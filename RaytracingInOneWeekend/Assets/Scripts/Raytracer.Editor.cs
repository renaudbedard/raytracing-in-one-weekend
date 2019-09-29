#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using System.IO;
using static Unity.Mathematics.math;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
#else
using OdinMock;
#endif

namespace RaytracerInOneWeekend
{
	partial class Raytracer
	{
		[ButtonGroup("Save")]
		[DisableInEditorMode]
		void SaveFrontBuffer()
		{
			byte[] pngBytes = frontBufferTexture.EncodeToPNG();
			File.WriteAllBytes(
				Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
					$"Raytracer {DateTime.Now:yyyy-MM-dd HH-mm-ss}.png"), pngBytes);
		}

		[ButtonGroup("Save")]
		[DisableInEditorMode]
		void SaveView()
		{
			ScreenCapture.CaptureScreenshot(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
				$"Raytracer {DateTime.Now:yyyy-MM-dd HH-mm-ss}.png"));
		}

		[DisableIf(nameof(TraceActive))]
		[DisableInEditorMode]
		[ButtonGroup("Trace")]
		void TriggerTrace() => ScheduleAccumulate(true);

		[EnableIf(nameof(TraceActive))]
		[DisableInEditorMode]
		[ButtonGroup("Trace")]
		void AbortTrace() => traceAborted = true;

#if BVH
		[SerializeField]
		[DisableInPlayMode]
		bool previewBvh = false;
#endif
		[SerializeField] [DisableInEditorMode] BufferView bufferView = BufferView.Front;

		CommandBuffer opaquePreviewCommandBuffer, transparentPreviewCommandBuffer;
		bool hookedEditorUpdate;

		MeshFilter previewSphere, previewRect, previewBox;

		[SerializeField] [HideInInspector]
		List<UnityEngine.Material> previewMaterials = new List<UnityEngine.Material>();

		void WatchForWorldChanges()
		{
			// watch for world data changes (won't catch those from OnValidate)
			if (scene && scene.Dirty)
			{
				Transform cameraTransform = targetCamera.transform;
				cameraTransform.position = scene.CameraPosition;
				cameraTransform.rotation = Quaternion.LookRotation(scene.CameraTarget - scene.CameraPosition);
				targetCamera.fieldOfView = scene.CameraFieldOfView;

				if (Application.isPlaying)
					worldNeedsRebuild = true;
				else
					EditorApplication.delayCall += UpdatePreview;

				scene.ClearDirty();
			}
		}


		void OnValidate()
		{
			if (Application.isPlaying)
			{
				if (commandBufferHooked)
				{
					targetCamera.RemoveCommandBuffer(CameraEvent.AfterEverything, commandBuffer);
					commandBufferHooked = false;
				}
			}
			else if (!EditorApplication.isPlayingOrWillChangePlaymode)
			{
				EditorApplication.delayCall += UpdatePreview;
				if (scene) scene.ClearDirty();
			}
		}

		void OnEditorUpdate()
		{
			if (!this)
			{
				// ReSharper disable once DelegateSubtraction
				EditorApplication.update -= OnEditorUpdate;
				return;
			}

			if (!EditorApplication.isPlaying)
			{
				transform.localPosition = Vector3.zero;
				WatchForWorldChanges();
			}

#if BVH
			if (bvhNodeBuffer.IsCreated && !EditorApplication.isPlaying &&
			    UnityEditorInternal.InternalEditorUtility.isApplicationActive)
			{
				EditorWindow.GetWindow<SceneView>().Repaint();
			}
#endif
		}

		void EnsurePreviewObjectExists(PrimitiveType type, ref MeshFilter previewObject)
		{
			if (previewObject != null) return;
			GameObject primitive = GameObject.CreatePrimitive(type);
			primitive.hideFlags = HideFlags.HideAndDontSave;
			primitive.SetActive(false);
			previewObject = primitive.GetComponent<MeshFilter>();
			if (type == PrimitiveType.Quad)
			{
				Mesh quadMesh = previewObject.sharedMesh;
				previewObject.sharedMesh = new Mesh
				{
					vertices = quadMesh.vertices,
					normals = quadMesh.normals.Select(x => -x).ToArray(),
					triangles = quadMesh.triangles.Reverse().ToArray(),
				};
			}
		}

		void UpdatePreview()
		{
			if (!this) return;
#if BVH
			if (previewBvh)
			{
				if (!entityBuffer.IsCreated) RebuildEntityBuffers();
				if (!bvhNodeBuffer.IsCreated) RebuildBvh();
			}
			else
			{
				sphereBuffer.SafeDispose();
				bvhNodeBuffer.SafeDispose();
				entityBuffer.SafeDispose();

				ActiveEntities.Clear();
			}
#endif // BVH

			if (scene)
			{
				Transform cameraTransform = targetCamera.transform;
				cameraTransform.position = scene.CameraPosition;
				cameraTransform.rotation = Quaternion.LookRotation(scene.CameraTarget - scene.CameraPosition);
				targetCamera.fieldOfView = scene.CameraFieldOfView;
			}

			Action updateDelegate = OnEditorUpdate;
			if (EditorApplication.update.GetInvocationList().All(x => x != (Delegate) updateDelegate))
				EditorApplication.update += OnEditorUpdate;

			if (opaquePreviewCommandBuffer == null)
				opaquePreviewCommandBuffer = new CommandBuffer { name = "World Preview (Opaque)" };
			if (transparentPreviewCommandBuffer == null)
				transparentPreviewCommandBuffer = new CommandBuffer { name = "World Preview (Transparent)" };

			targetCamera.RemoveAllCommandBuffers();

			targetCamera.AddCommandBuffer(CameraEvent.AfterForwardOpaque, opaquePreviewCommandBuffer);
			targetCamera.AddCommandBuffer(CameraEvent.AfterForwardAlpha, transparentPreviewCommandBuffer);

			opaquePreviewCommandBuffer.Clear();
			transparentPreviewCommandBuffer.Clear();

			foreach (UnityEngine.Material material in previewMaterials)
				DestroyImmediate(material);
			previewMaterials.Clear();

			var skybox = targetCamera.GetComponent<Skybox>();
			if (AssetDatabase.IsMainAsset(skybox.material))
				RenderSettings.skybox = skybox.material = new UnityEngine.Material(skybox.material);

			skybox.material.SetColor("_Color1", scene.SkyBottomColor);
			skybox.material.SetColor("_Color2", scene.SkyTopColor);

			EnsurePreviewObjectExists(PrimitiveType.Sphere, ref previewSphere);
			EnsurePreviewObjectExists(PrimitiveType.Quad, ref previewRect);
			EnsurePreviewObjectExists(PrimitiveType.Cube, ref previewBox);

			var previewMeshRenderer = previewSphere.GetComponent<MeshRenderer>();

			CollectActiveEntities();

			opaquePreviewCommandBuffer.EnableShaderKeyword("LIGHTPROBE_SH");
			transparentPreviewCommandBuffer.EnableShaderKeyword("LIGHTPROBE_SH");

			foreach (EntityData entity in ActiveEntities)
			{
				if (!entity.Material) continue;

				bool transparent = entity.Material.Type == MaterialType.Dielectric;
				bool emissive = entity.Material.Type == MaterialType.DiffuseLight;

				Color albedoMainColor = entity.Material.Albedo ? entity.Material.Albedo.MainColor : Color.white;
				Color color = transparent ? albedoMainColor.GetAlphaReplaced(0.5f) : albedoMainColor;
				var material = new UnityEngine.Material(previewMeshRenderer.sharedMaterial) { color = color };
				previewMaterials.Add(material);

				material.SetFloat("_Metallic", entity.Material.Type == MaterialType.Metal ? 1 : 0);
				material.SetFloat("_Glossiness",
					entity.Material.Type == MaterialType.Metal ? 1 - entity.Material.Fuzz : transparent ? 1 : 0);
				material.SetTexture("_MainTex", entity.Material.Albedo ? entity.Material.Albedo.Image : null);

				if (transparent)
				{
					material.SetInt("_SrcBlend", (int) BlendMode.One);
					material.SetInt("_DstBlend", (int) BlendMode.OneMinusSrcAlpha);
					material.SetInt("_ZWrite", 0);
					material.EnableKeyword("_ALPHAPREMULTIPLY_ON");
					material.renderQueue = 3000;
				}

				if (emissive)
				{
					material.EnableKeyword("_EMISSION");
					material.SetColor("_EmissionColor",
						entity.Material.Emission ? entity.Material.Emission.MainColor : Color.black);
				}

				CommandBuffer previewCommandBuffer =
					transparent ? transparentPreviewCommandBuffer : opaquePreviewCommandBuffer;

				switch (entity.Type)
				{
					case EntityType.Sphere:
						SphereData s = entity.SphereData;
						previewCommandBuffer.DrawMesh(previewSphere.sharedMesh,
							Matrix4x4.TRS(entity.Position, entity.Rotation, s.Radius * 2 * Vector3.one),
							material, 0,
							material.FindPass("FORWARD"));
						break;

					case EntityType.Rect:
						RectData r = entity.RectData;
						previewCommandBuffer.DrawMesh(previewRect.sharedMesh,
							Matrix4x4.TRS(entity.Position, Quaternion.LookRotation(Vector3.forward) * entity.Rotation,float3(r.Size, 1)),
							material, 0,
							material.FindPass("FORWARD"));
						break;

					case EntityType.Box:
						BoxData b = entity.BoxData;
						previewCommandBuffer.DrawMesh(previewBox.sharedMesh,
							Matrix4x4.TRS(entity.Position, entity.Rotation, b.Size), material, 0,
							material.FindPass("FORWARD"));
						break;
				}
			}
		}

		void ForceUpdateInspector()
		{
			EditorUtility.SetDirty(this);
		}

#if BVH
		unsafe
#endif
		void OnDrawGizmos()
		{
			var sceneCameraTransform = SceneView.GetAllSceneCameras()[0].transform;

			EnsurePreviewObjectExists(PrimitiveType.Quad, ref previewRect);

			foreach (EntityData e in ActiveEntities
				.Where(x => x.Material)
				.OrderBy(x => Vector3.Dot(sceneCameraTransform.position - x.Position, sceneCameraTransform.forward)))
			{
				Color color = e.Material.Albedo ? e.Material.Albedo.MainColor : Color.white;
				if (e.Material.Emission)
					color += e.Material.Emission.MainColor;

				switch (e.Material.Type)
				{
					case MaterialType.Dielectric: color = color.GetAlphaReplaced(0.5f); break;
					case MaterialType.Isotropic: color = color.GetAlphaReplaced(max(e.Material.Density, 0.5f)); break;
					default: color = color.GetAlphaReplaced(1); break;
				}

				Gizmos.color = color;

				if (e.Selected)
					Gizmos.color = Color.yellow;

				switch (e.Type)
				{
					case EntityType.Rect:
						Gizmos.matrix = Matrix4x4.TRS(e.Position, e.Rotation, float3(e.RectData.Size, 1));
						Gizmos.DrawMesh(previewRect.sharedMesh, 0);
						Gizmos.matrix = Matrix4x4.identity;
						break;

					case EntityType.Sphere:
						Gizmos.DrawSphere(e.Position, e.SphereData.Radius);
						break;

					case EntityType.Box:
						Gizmos.matrix = Matrix4x4.TRS(e.Position, e.Rotation, Vector3.one);
						Gizmos.DrawCube(Vector3.zero, e.BoxData.Size);
						Gizmos.matrix = Matrix4x4.identity;
						break;
				}
			}

#if BVH
			if (previewBvh && bvhNodeBuffer.IsCreated)
			{
				float silverRatio = (sqrt(5.0f) - 1.0f) / 2.0f;
				(AxisAlignedBoundingBox _, int Depth)[] subBounds = BvhRoot->GetAllSubBounds().ToArray();
				int maxDepth = subBounds.Max(x => x.Depth);
				int shownLayer = DateTime.Now.Second % (maxDepth + 1);
				int i = -1;
				foreach ((AxisAlignedBoundingBox bounds, int depth) in subBounds)
				{
					i++;
					if (depth != shownLayer) continue;

					Gizmos.color = Color.HSVToRGB(frac(i * silverRatio), 1, 1).GetAlphaReplaced(0.6f);
					Gizmos.DrawCube(bounds.Center, bounds.Size);
				}
			}
#endif

#if PATH_DEBUGGING
			if (debugPaths.IsCreated)
			{
				float alpha = 1;
				foreach (DebugPath path in debugPaths)
				{
					Debug.DrawLine(path.From, path.To,
						fadeDebugPaths ? Color.white.GetAlphaReplaced(alpha) : Color.white, debugPathDuration);
					alpha *= 0.5f;
				}
			}
#endif
		}
	}
}
#endif // UNITY_EDITOR