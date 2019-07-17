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

#if UNITY_EDITOR
		void OnValidate()
		{
			if (Application.isPlaying || UnityEditor.EditorApplication.isCompiling)
				return;

			var definitions = new List<string>();
#if ODIN_INSPECTOR
			definitions.Add("ODIN_INSPECTOR");
#endif
			switch (dataLayout)
			{
				case DataLayout.StructOfArrays:
					definitions.Add("MANUAL_SOA");
					break;

				case DataLayout.Interleaved:
					definitions.Add("MANUAL_AOSOA");
					break;

				case DataLayout.AutomaticSOA:
					definitions.Add("UNITY_SOA");
					materialStorage = MaterialStorage.Buffered;
					break;
			}

			switch (materialStorage)
			{
				case MaterialStorage.Buffered:
					definitions.Add("BUFFERED_MATERIALS");
					break;
			}

			UnityEditor.PlayerSettings.SetScriptingDefineSymbolsForGroup(
				UnityEditor.BuildTargetGroup.Standalone,
				string.Join(";", definitions));
		}
#endif
	}
}