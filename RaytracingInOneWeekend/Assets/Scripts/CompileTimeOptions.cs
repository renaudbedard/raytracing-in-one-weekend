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
		enum DataLayout
		{
			ArrayOfStructs,
			StructOfArrays,
			Interleaved,
			AutomaticSOA
		}

		enum MaterialStorage
		{
			Inline,
			Buffered
		}

		enum SpacePartitioning
		{
			None,
			BVH
		}

		[SerializeField]
#if ODIN_INSPECTOR
		[DisableInPlayMode]
#endif
		DataLayout dataLayout = DataLayout.StructOfArrays;

		[SerializeField]
#if ODIN_INSPECTOR
		[DisableInPlayMode] [DisableIf(nameof(dataLayout), DataLayout.AutomaticSOA)]
#endif
		MaterialStorage materialStorage = MaterialStorage.Inline;

		[SerializeField]
#if ODIN_INSPECTOR
		[DisableInPlayMode]
#endif
		SpacePartitioning spacePartitioning = SpacePartitioning.BVH;

#if UNITY_EDITOR
		void OnValidate()
		{
			if (Application.isPlaying || UnityEditor.EditorApplication.isCompiling)
				return;

			string currentDefines =
				UnityEditor.PlayerSettings.GetScriptingDefineSymbolsForGroup(UnityEditor.BuildTargetGroup.Standalone);

			var originalDefinitions = new HashSet<string>(currentDefines.Split(';'));
			var newDefinitions = new HashSet<string>(originalDefinitions);

			newDefinitions.Remove("MANUAL_SOA");
			newDefinitions.Remove("MANUAL_AOSOA");
			newDefinitions.Remove("UNITY_SOA");
			newDefinitions.Remove("BUFFERED_MATERIALS");
			newDefinitions.Remove("BVH");

			switch (dataLayout)
			{
				case DataLayout.StructOfArrays:
					newDefinitions.Add("MANUAL_SOA");
					break;

				case DataLayout.Interleaved:
					newDefinitions.Add("MANUAL_AOSOA");
					break;

				case DataLayout.AutomaticSOA:
					newDefinitions.Add("UNITY_SOA");
					materialStorage = MaterialStorage.Buffered;
					break;
			}

			switch (materialStorage)
			{
				case MaterialStorage.Buffered:
					newDefinitions.Add("BUFFERED_MATERIALS");
					break;
			}

			switch (spacePartitioning)
			{
				case SpacePartitioning.BVH:
					newDefinitions.Add("BVH");
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