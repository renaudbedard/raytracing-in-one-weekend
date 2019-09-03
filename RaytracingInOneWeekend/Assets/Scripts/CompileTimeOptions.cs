using System.Collections.Generic;
using UnityEngine;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
#else
using OdinMock;
#endif

namespace RaytracerInOneWeekend
{
	[ExecuteInEditMode]
	class CompileTimeOptions : MonoBehaviour
	{
		enum HitTestingMode
		{
			Basic = 0,
			RecursiveBvh = 3,
			IterativeBvh = 4,
			//IterativeBvhSimd = 5
		}

		[SerializeField] [DisableInPlayMode] HitTestingMode hitTestingMode = HitTestingMode.Basic;
		[SerializeField] [DisableInPlayMode] bool fullDiagnostics = false, pathDebugging = false;

#if UNITY_EDITOR
		void OnValidate()
		{
			if (Application.isPlaying || UnityEditor.EditorApplication.isCompiling)
				return;

			string currentDefines =
				UnityEditor.PlayerSettings.GetScriptingDefineSymbolsForGroup(UnityEditor.BuildTargetGroup.Standalone);

			var originalDefinitions = new HashSet<string>(currentDefines.Split(';'));
			var newDefinitions = new HashSet<string>(originalDefinitions);

			newDefinitions.Remove("BASIC");
			newDefinitions.Remove("BUFFERED_MATERIALS");
			newDefinitions.Remove("BVH");
			newDefinitions.Remove("BVH_RECURSIVE");
			newDefinitions.Remove("BVH_ITERATIVE");
			newDefinitions.Remove("BVH_SIMD");
			newDefinitions.Remove("QUAD_BVH");
			newDefinitions.Remove("FULL_DIAGNOSTICS");
			newDefinitions.Remove("PATH_DEBUGGING");

			switch (hitTestingMode)
			{
				case HitTestingMode.Basic:
					newDefinitions.Add("BASIC");
					break;

				case HitTestingMode.RecursiveBvh:
					newDefinitions.Add("BVH");
					newDefinitions.Add("BVH_RECURSIVE");
					break;

				case HitTestingMode.IterativeBvh:
					newDefinitions.Add("BVH");
					newDefinitions.Add("BVH_ITERATIVE");
					break;

				// case HitTestingMode.IterativeBvhSimd:
				// 	newDefinitions.Add("BVH");
				// 	newDefinitions.Add("BVH_ITERATIVE");
				// 	newDefinitions.Add("BVH_SIMD");
				// 	break;
			}

			if (fullDiagnostics) newDefinitions.Add("FULL_DIAGNOSTICS");
			if (pathDebugging) newDefinitions.Add("PATH_DEBUGGING");

			if (!newDefinitions.SetEquals(originalDefinitions))
			{
				UnityEditor.EditorApplication.delayCall += () =>
					UnityEditor.PlayerSettings.SetScriptingDefineSymbolsForGroup(
						UnityEditor.BuildTargetGroup.Standalone,
						string.Join(";", newDefinitions));
			}
		}
#endif
	}
}