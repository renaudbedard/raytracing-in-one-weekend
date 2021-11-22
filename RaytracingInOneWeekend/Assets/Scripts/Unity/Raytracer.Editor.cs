#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using Runtime;
using UnityEditor;
using UnityEngine;
using Util;
using static Unity.Mathematics.math;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
#else
using OdinMock;
#endif

namespace Unity
{
	partial class Raytracer
	{
		[ButtonGroup("Save")]
		[DisableInEditorMode]
		[UsedImplicitly]
		void SaveFrontBuffer()
		{
			byte[] pngBytes = frontBufferTexture.EncodeToPNG();
			File.WriteAllBytes(Path.Combine(System.Environment.GetFolderPath(System.Environment.SpecialFolder.Desktop), $"Raytracer {DateTime.Now:yyyy-MM-dd HH-mm-ss}.png"), pngBytes);
		}

		[ButtonGroup("Save")]
		[DisableInEditorMode]
		[UsedImplicitly]
		void SaveView()
		{
			ScreenCapture.CaptureScreenshot(Path.Combine(System.Environment.GetFolderPath(System.Environment.SpecialFolder.Desktop), $"Raytracer {DateTime.Now:yyyy-MM-dd HH-mm-ss}.png"));
		}

		[DisableIf(nameof(TraceActive))]
		[DisableInEditorMode]
		[ButtonGroup("Trace")]
		[UsedImplicitly]
		void TriggerTrace() => ScheduleSample(true);

		[EnableIf(nameof(TraceActive))]
		[DisableInEditorMode]
		[ButtonGroup("Trace")]
		[UsedImplicitly]
		void AbortTrace() => traceAborted = true;

		[SerializeField] bool debugFailedSamples = false;
		[SerializeField] bool previewBvh = false;
		[SerializeField] [DisableInEditorMode] BufferView bufferView = BufferView.Front;

		[InitializeOnLoadMethod]
		static void Construct()
		{
			ObjectChangeEvents.changesPublished += OnChangesPublished;
		}

		static void OnChangesPublished(ref ObjectChangeEventStream stream)
		{
			if (!Application.isPlaying)
				return;

			var raytracer = FindObjectOfType<Raytracer>();
			if (!raytracer)
				return;

			for (int i = 0; i < stream.length; i++)
			{
				var eventType = stream.GetEventType(i);
				switch (eventType)
				{
					case ObjectChangeKind.ChangeGameObjectOrComponentProperties:
						stream.GetChangeGameObjectOrComponentPropertiesEvent(i, out var data);
						var modifiedObject = EditorUtility.InstanceIDToObject(data.instanceId);
						if (modifiedObject == raytracer)
							continue;
						raytracer.worldNeedsRebuild = true;
						break;

					case ObjectChangeKind.ChangeGameObjectParent:
					case ObjectChangeKind.ChangeGameObjectStructure:
					case ObjectChangeKind.CreateGameObjectHierarchy:
					case ObjectChangeKind.ChangeGameObjectStructureHierarchy:
					case ObjectChangeKind.UpdatePrefabInstances:
					case ObjectChangeKind.ChangeAssetObjectProperties:
					case ObjectChangeKind.CreateAssetObject:
					case ObjectChangeKind.DestroyAssetObject:
						raytracer.worldNeedsRebuild = true;
						break;
				}
			}
		}

		void ForceUpdateInspector()
		{
			EditorUtility.SetDirty(this);
		}

		void OnDrawGizmos()
		{
			if (previewBvh)
			{
				if (!BvhRootData.HasValue)
				{
					RebuildEntityBuffers();
					RebuildBvh(true);
				}

				float silverRatio = (sqrt(5.0f) - 1.0f) / 2.0f;
				(AxisAlignedBoundingBox _, int Depth)[] subBounds = BvhRootData.Value.GetAllSubBounds().ToArray();
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

				SceneView.RepaintAll();
			}

#if PATH_DEBUGGING
			if (debugPaths.IsCreated)
			{
				float silverRatio = (sqrt(5.0f) - 1.0f) / 2.0f;
				float alpha = 1;
				uint i = frameSeed;
				foreach (DebugPath path in debugPaths)
				{
					var color = Color.HSVToRGB(frac(i * silverRatio), 1, 1);
					Debug.DrawLine(path.From, path.To, fadeDebugPaths ? color.GetAlphaReplaced(alpha) : color, debugPathDuration);
					alpha *= 0.5f;
					i++;
				}
			}
#endif
		}
	}
}
#endif // UNITY_EDITOR