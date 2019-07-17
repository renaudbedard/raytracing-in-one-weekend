using JetBrains.Annotations;
using UnityEngine;

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
		[Title("Debug")]
		[UsedImplicitly]
#if ODIN_INSPECTOR
		[ShowInInspector] [ReadOnly]
#else
		public
#endif
		uint accumulatedSamples;

		[UsedImplicitly]
#if ODIN_INSPECTOR
		[ShowInInspector] [ReadOnly]
#else
		public
#endif
		float millionRaysPerSecond, avgMRaysPerSecond, lastBatchDuration, lastTraceDuration;

#if ODIN_INSPECTOR
		[ShowInInspector] [InlineEditor(InlineEditorModes.LargePreview)] [ReadOnly]
#else
		public
#endif
		Texture2D frontBufferTexture;

#if ODIN_INSPECTOR
		[DisableIf(nameof(TraceActive))] [DisableInEditorMode] [Button]
		void TriggerTrace() => ScheduleAccumulate(true);

		[EnableIf(nameof(TraceActive))] [DisableInEditorMode] [Button]
		void AbortTrace() => traceAborted = true;
#endif

#if UNITY_EDITOR
		void OnValidate()
		{
			if (Application.isPlaying)
				worldNeedsRebuild = true;
		}
#endif

		void ForceUpdateInspector()
		{
#if UNITY_EDITOR
			UnityEditor.EditorUtility.SetDirty(this);
#endif
		}

#if ODIN_INSPECTOR && UNITY_EDITOR
		[Button] [DisableInEditorMode]
		void SaveFrontBuffer()
		{
			byte[] pngBytes = frontBufferTexture.EncodeToPNG();
			File.WriteAllBytes(
				Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
					$"Raytracer {DateTime.Now:yyyy-MM-dd HH-mm-ss}.png"), pngBytes);
		}
#endif
	}
}