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
	[InitializeOnLoad]
	partial class Raytracer
	{
		static Raytracer()
		{
			EditorApplication.update += () =>
			{
				if (EditorApplication.isPlaying) return;
				foreach (var raytracer in FindObjectsOfType<Raytracer>())
					raytracer.WatchForDirtyMaterials();
			};
		}

#if ODIN_INSPECTOR
		[DisableIf(nameof(TraceActive))]
		[DisableInEditorMode]
		[Button]
		void TriggerTrace() => ScheduleAccumulate(true);

		[EnableIf(nameof(TraceActive))]
		[DisableInEditorMode]
		[Button]
		void AbortTrace() => traceAborted = true;
#endif

		CommandBuffer previewCommandBuffer;
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
			if (previewCommandBuffer == null)
				previewCommandBuffer = new CommandBuffer { name = "World Preview" };
			targetCamera.RemoveAllCommandBuffers();
			targetCamera.AddCommandBuffer(CameraEvent.AfterForwardOpaque, previewCommandBuffer);

			previewCommandBuffer.Clear();

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
			foreach (SphereData sphere in activeSpheres)
			{
				sphere.Material.Dirty = false;

				UnityEngine.Material material = new UnityEngine.Material(meshRenderer.sharedMaterial)
				{
					color = sphere.Material.Albedo
				};
				material.SetFloat("_Metallic", sphere.Material.Type == MaterialType.Metal ? 1 : 0);
				material.SetFloat("_Glossiness",
					sphere.Material.Type == MaterialType.Metal ? 1 - sphere.Material.Fuzz :
					sphere.Material.Type == MaterialType.Dielectric ? 1 : 0);

				previewCommandBuffer.EnableShaderKeyword("LIGHTPROBE_SH");
				previewCommandBuffer.DrawMesh(meshFilter.sharedMesh,
					Matrix4x4.TRS(sphere.Center, Quaternion.identity, sphere.Radius * 2 * Vector3.one), material, 0,
					material.FindPass("FORWARD"));
				previewMaterials.Add(material);
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
					? new Color(albedo.r, albedo.g, albedo.b, 0.5f)
					: albedo;

				Gizmos.DrawSphere(sphere.Center, sphere.Radius);
			}
		}
	}
}
#endif