using System.Collections.Generic;
using Sirenix.OdinInspector;
using UnityEngine;
using Unity.Mathematics;
using UnityEditor;
using UnityEngine.Rendering;
using static Unity.Mathematics.math;

namespace Unity
{
	public class GridGenerator : MonoBehaviour
	{
		const string GeneratedName = "Generated Instance";

		[SerializeField] MeshRenderer template;
		[SerializeField] int2 gridSize;
		[SerializeField] Transform minimumTransform, maximumTransform;

		[Header("Horizontal")]
		[SerializeField] Material horizontalMinimumMaterial;
		[SerializeField] Material horizontalMaximumMaterial;
		[SerializeField] [ValueDropdown(nameof(MaterialPropertiesNames))] int horizontalBlendParameter;

		[Header("Vertical")]
		[SerializeField] Material verticalMinimumMaterial;
		[SerializeField] Material verticalMaximumMaterial;
		[SerializeField] [ValueDropdown(nameof(MaterialPropertiesNames))] int verticalBlendParameter;

		readonly List<GameObject> instances = new();

		IEnumerable<ValueDropdownItem> MaterialPropertiesNames
		{
			get
			{
				for (int i = 0; i < horizontalMinimumMaterial.shader.GetPropertyCount(); i++)
					yield return new ValueDropdownItem(horizontalMinimumMaterial.shader.GetPropertyName(i), i);
			}
		}

		void OnValidate()
		{
			if (EditorApplication.isPlayingOrWillChangePlaymode)
				return;

			Debug.Log("Regenerating grid...");

			instances.Clear();

			foreach (Transform instance in transform)
				if (instance.name == GeneratedName)
					instances.Add(instance.gameObject);

			foreach (GameObject instance in instances)
				EditorApplication.delayCall += () =>
				{
					DestroyImmediate(instance.GetComponent<Renderer>().sharedMaterial);
					DestroyImmediate(instance);
				};

			instances.Clear();

			float3 minPosition = minimumTransform.localPosition;
			float3 maxPosition = maximumTransform.localPosition;

			float3 horizontalAxis = Mathf.Approximately(minPosition.x, maxPosition.x) ? float3(0, 0, 1) : float3(1, 0, 0);
			float3 verticalAxis = 1 - horizontalAxis;

			float3 delta = maxPosition - minPosition;

			for (int j = 0; j < gridSize.y; j++)
			for (int i = 0; i < gridSize.x; i++)
			{
				var hs = (float)i / (gridSize.x - 1);
				var vs = (float)j / (gridSize.y - 1);

				MeshRenderer instance = Instantiate(template, transform);
				instance.transform.localPosition = dot(delta, horizontalAxis) * hs * horizontalAxis * sign(delta) +
				                                   dot(delta, verticalAxis) * vs * verticalAxis * sign(delta) +
				                                   minPosition;
				instance.name = GeneratedName;

				var materialInstance = new Material(horizontalMinimumMaterial) { name = "Generated Material" };

				switch (materialInstance.shader.GetPropertyType(horizontalBlendParameter))
				{
					case ShaderPropertyType.Float:
					case ShaderPropertyType.Range:
						int nameId = materialInstance.shader.GetPropertyNameId(horizontalBlendParameter);
						materialInstance.SetFloat(nameId, lerp(horizontalMinimumMaterial.GetFloat(nameId), horizontalMaximumMaterial.GetFloat(nameId), hs));
						break;
				}

				switch (materialInstance.shader.GetPropertyType(verticalBlendParameter))
				{
					case ShaderPropertyType.Float:
					case ShaderPropertyType.Range:
						int nameId = materialInstance.shader.GetPropertyNameId(verticalBlendParameter);
						materialInstance.SetFloat(nameId, lerp(verticalMinimumMaterial.GetFloat(nameId), verticalMaximumMaterial.GetFloat(nameId), vs));
						break;
				}

				instance.material = materialInstance;
			}
		}
	}
}