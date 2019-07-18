
using Unity.Mathematics;
#if UNITY_EDITOR
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

#if ODIN_INSPECTOR
using System;
using Sirenix.OdinInspector;
using System.IO;
#else
using Title = UnityEngine.HeaderAttribute;
#endif

namespace RaytracerInOneWeekend
{
	partial class Raytracer
	{
#if ODIN_INSPECTOR
		[DisableIf(nameof(TraceActive))]
		[DisableInEditorMode]
		[Button]
		void TriggerTrace() => ScheduleAccumulate(true);

		[EnableIf(nameof(TraceActive))]
		[DisableInEditorMode]
		[Button]
		void AbortTrace() => traceAborted = true;

		[DisableIf(nameof(TraceActive))]
		[Button]
		void ForceRebuildBVH()
		{
			RebuildEntityBuffer();
			RebuildBvh();
		}
#endif

		CommandBuffer opaquePreviewCommandBuffer, transparentPreviewCommandBuffer;
		bool hookedEditorUpdate;

		[SerializeField] [HideInInspector] GameObject previewObject;
		[SerializeField] [HideInInspector] List<UnityEngine.Material> previewMaterials = new List<UnityEngine.Material>();

		void WatchForDirtyMaterials()
		{
			// watch for material data changes (won't catch those from OnValidate)
			if (!randomScene && spheres.Any(x => x.Material.Dirty))
			{
				if (Application.isPlaying)
					worldNeedsRebuild = true;
				else
					EditorApplication.delayCall += UpdatePreview;
			}
		}

		void OnValidate()
		{
			if (Application.isPlaying)
				worldNeedsRebuild = true;
			else if (!EditorApplication.isPlayingOrWillChangePlaymode)
				EditorApplication.delayCall += UpdatePreview;
		}

		void UpdatePreview()
		{
			if (!hookedEditorUpdate)
			{
				EditorApplication.update += () =>
				{
					if (!EditorApplication.isPlaying)
						WatchForDirtyMaterials();
				};
				hookedEditorUpdate = true;
			}

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
			skybox.material.SetColor("_Color1", skyBottomColor);
			skybox.material.SetColor("_Color2", skyTopColor);

			if (previewObject == null)
				previewObject = GameObject.CreatePrimitive(UnityEngine.PrimitiveType.Sphere);
			previewObject.hideFlags = HideFlags.HideAndDontSave;
			previewObject.SetActive(false);

			var meshRenderer = previewObject.GetComponent<MeshRenderer>();
			var meshFilter = previewObject.GetComponent<MeshFilter>();

			CollectActiveSpheres();

			opaquePreviewCommandBuffer.EnableShaderKeyword("LIGHTPROBE_SH");
			transparentPreviewCommandBuffer.EnableShaderKeyword("LIGHTPROBE_SH");

			foreach (SphereData sphere in activeSpheres)
			{
				sphere.Material.Dirty = false;

				bool transparent = sphere.Material.Type == MaterialType.Dielectric;

				Color color = transparent ? sphere.Material.Albedo.GetAlphaReplaced(0.5f) : sphere.Material.Albedo;
				var material = new UnityEngine.Material(meshRenderer.sharedMaterial) { color = color };
				previewMaterials.Add(material);

				material.SetFloat("_Metallic", sphere.Material.Type == MaterialType.Metal ? 1 : 0);
				material.SetFloat("_Glossiness",
					sphere.Material.Type == MaterialType.Metal ? 1 - sphere.Material.Fuzz : transparent ? 1 : 0);

				if (transparent)
				{
					material.SetInt("_SrcBlend", (int) BlendMode.One);
					material.SetInt("_DstBlend", (int) BlendMode.OneMinusSrcAlpha);
					material.SetInt("_ZWrite", 0);
					material.EnableKeyword("_ALPHAPREMULTIPLY_ON");
					material.renderQueue = 3000;
				}

				CommandBuffer previewCommandBuffer =
					transparent ? transparentPreviewCommandBuffer : opaquePreviewCommandBuffer;

				previewCommandBuffer.DrawMesh(meshFilter.sharedMesh,
					Matrix4x4.TRS(sphere.Center, Quaternion.identity, sphere.Radius * 2 * Vector3.one), material, 0,
					material.FindPass("FORWARD"));
			}
		}

		void ForceUpdateInspector()
		{
			EditorUtility.SetDirty(this);
		}

#if ODIN_INSPECTOR
		[Button]
		[DisableInEditorMode]
		void SaveFrontBuffer()
		{
			byte[] pngBytes = frontBufferTexture.EncodeToPNG();
			File.WriteAllBytes(
				Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
					$"Raytracer {DateTime.Now:yyyy-MM-dd HH-mm-ss}.png"), pngBytes);
		}
#endif

		void OnDrawGizmos()
		{
			var sceneCameraTransform = SceneView.GetAllSceneCameras()[0].transform;
			foreach (SphereData sphere in activeSpheres
				.OrderBy(x => Vector3.Dot(sceneCameraTransform.position - x.Center, sceneCameraTransform.forward)))
			{
				Color albedo = sphere.Material.Albedo;
				Gizmos.color = sphere.Material.Type == MaterialType.Dielectric
					? albedo.GetAlphaReplaced(0.5f)
					: albedo;

				Gizmos.DrawSphere(sphere.Center, sphere.Radius);
			}

#if BVH
			foreach (BvhNode node in World)
			{
				Gizmos.color = Color.red.GetAlphaReplaced(0.25f);
				float3 size = node.Bounds.Max - node.Bounds.Min;
				Gizmos.DrawCube(node.Bounds.Min + size / 2, size);
			}
#endif
		}
	}
}
#endif