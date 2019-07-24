using System.Collections.Generic;
using UnityEngine;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
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

		enum MaterialStorage
		{
			Inline,
			Buffered
		}

		[SerializeField]
#if ODIN_INSPECTOR
		[DisableInPlayMode]
#endif
		HitTestingMode hitTestingMode = HitTestingMode.Basic;

		[SerializeField]
#if ODIN_INSPECTOR
		[DisableInPlayMode] [DisableIf(nameof(dataLayout), DataLayout.AutomaticSOA)]
#endif
		MaterialStorage materialStorage = MaterialStorage.Inline;
		
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

			switch (materialStorage)
			{
				case MaterialStorage.Buffered:
					newDefinitions.Add("BUFFERED_MATERIALS");
					break;
			}

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