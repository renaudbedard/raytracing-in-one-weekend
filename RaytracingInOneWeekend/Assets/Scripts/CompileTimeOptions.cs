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
			Basic,
			SoaSimd,
			AosoaSimd,
			RecursiveBvh,
			IterativeBvh,
			IterativeBvhSimd
		}

		[SerializeField]
		[DisableInPlayMode]
		HitTestingMode hitTestingMode = HitTestingMode.Basic;

		[SerializeField]
		[DisableInPlayMode]
		bool fullDiagnostics = false;

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
			newDefinitions.Remove("SOA_SIMD");
			newDefinitions.Remove("AOSOA_SIMD");
			newDefinitions.Remove("BUFFERED_MATERIALS");
			newDefinitions.Remove("BVH");
			newDefinitions.Remove("BVH_RECURSIVE");
			newDefinitions.Remove("BVH_ITERATIVE");
			newDefinitions.Remove("BVH_SIMD");
			newDefinitions.Remove("QUAD_BVH");
			newDefinitions.Remove("FULL_DIAGNOSTICS");

			switch (hitTestingMode)
			{
				case HitTestingMode.Basic:
					newDefinitions.Add("BASIC");
					break;

				case HitTestingMode.SoaSimd:
					newDefinitions.Add("SOA_SIMD");
					break;

				case HitTestingMode.AosoaSimd:
					newDefinitions.Add("AOSOA_SIMD");
					break;

				case HitTestingMode.RecursiveBvh:
					newDefinitions.Add("BVH");
					newDefinitions.Add("BVH_RECURSIVE");
					break;

				case HitTestingMode.IterativeBvh:
					newDefinitions.Add("BVH");
					newDefinitions.Add("BVH_ITERATIVE");
					break;

				case HitTestingMode.IterativeBvhSimd:
					newDefinitions.Add("BVH");
					newDefinitions.Add("BVH_ITERATIVE");
					newDefinitions.Add("BVH_SIMD");
					break;
			}

			if (fullDiagnostics)
				newDefinitions.Add("FULL_DIAGNOSTICS");

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