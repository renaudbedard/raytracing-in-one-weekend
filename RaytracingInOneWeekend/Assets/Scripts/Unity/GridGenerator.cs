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

		[Header("Bottom")]
		[SerializeField] Material bottomLeftMaterial;
		[SerializeField] Material bottomRightMaterial;

		[Header("Top")]
		[SerializeField] Material topLeftMaterial;
		[SerializeField] Material topRightMaterial;

		[Header("Blend Parameters")]
		[SerializeField] [ValueDropdown(nameof(MaterialPropertiesNames))] int verticalBlendParameter;
		[SerializeField] [ValueDropdown(nameof(MaterialPropertiesNames))] int horizontalBlendParameter;

		readonly List<GameObject> instances = new();

		IEnumerable<ValueDropdownItem> MaterialPropertiesNames
		{
			get
			{
				for (int i = 0; i < bottomLeftMaterial.shader.GetPropertyCount(); i++)
					yield return new ValueDropdownItem(bottomLeftMaterial.shader.GetPropertyName(i), i);
			}
		}

		void OnValidate()
		{
			if (EditorApplication.isPlayingOrWillChangePlaymode)
				return;

			EditorApplication.delayCall += Regenerate;
		}

		[Button("Regenerate")]
		void Regenerate()
		{
			if (this == null)
				return;

			instances.Clear();

			foreach (Transform instance in transform)
				if (instance.name == GeneratedName)
					instances.Add(instance.gameObject);

			foreach (GameObject instance in instances)
			{
				DestroyImmediate(instance.GetComponent<Renderer>().sharedMaterial);
				DestroyImmediate(instance);
			}

			instances.Clear();

			float3 minPosition = minimumTransform.localPosition;
			float3 maxPosition = maximumTransform.localPosition;

			float3 horizontalAxis = Mathf.Approximately(minPosition.x, maxPosition.x) ? float3(0, 0, 1) : float3(1, 0, 0);
			float3 verticalAxis = 1 - horizontalAxis;

			float3 delta = maxPosition - minPosition;

			for (int j = 0; j < gridSize.y; j++)
			for (int i = 0; i < gridSize.x; i++)
			{
				var hs = gridSize.x == 1 ? 0 : (float)i / (gridSize.x - 1);
				var vs = gridSize.y == 1 ? 0 : (float)j / (gridSize.y - 1);

				MeshRenderer instance = Instantiate(template, transform);
				instance.transform.localPosition = dot(delta, horizontalAxis) * hs * horizontalAxis * sign(delta) +
				                                   dot(delta, verticalAxis) * vs * verticalAxis * sign(delta) +
				                                   minPosition;
				instance.name = GeneratedName;

				var materialInstance = new Material(bottomLeftMaterial) { name = "Generated Material" };

				if (horizontalBlendParameter == verticalBlendParameter)
				{
					var blendParameter = verticalBlendParameter; // interchangeable with the bottom one
					int nameId = materialInstance.shader.GetPropertyNameId(blendParameter);

					switch (materialInstance.shader.GetPropertyType(blendParameter))
					{
						case ShaderPropertyType.Float:
						case ShaderPropertyType.Range:
						{
							float bottomFloatValue = lerp(bottomLeftMaterial.GetFloat(nameId), bottomRightMaterial.GetFloat(nameId), hs);
							float topFloatValue = lerp(topLeftMaterial.GetFloat(nameId), topRightMaterial.GetFloat(nameId), hs);
							materialInstance.SetFloat(nameId, lerp(bottomFloatValue, topFloatValue, vs));
							break;
						}

						case ShaderPropertyType.Color:
						{
							Color bottomColorValue = Color.Lerp(bottomLeftMaterial.GetColor(nameId), bottomRightMaterial.GetColor(nameId), hs);
							Color topColorValue = Color.Lerp(topLeftMaterial.GetColor(nameId), topRightMaterial.GetColor(nameId), hs);
							materialInstance.SetColor(nameId, Color.Lerp(bottomColorValue, topColorValue, vs));
							break;
						}
					}
				}
				else
				{
					int nameId = materialInstance.shader.GetPropertyNameId(horizontalBlendParameter);
					switch (materialInstance.shader.GetPropertyType(horizontalBlendParameter))
					{
						case ShaderPropertyType.Float:
						case ShaderPropertyType.Range:
						{
							float horizontalValue = lerp(bottomLeftMaterial.GetFloat(nameId), bottomRightMaterial.GetFloat(nameId), hs);
							materialInstance.SetFloat(nameId, horizontalValue);
							break;
						}

						case ShaderPropertyType.Color:
						{
							Color horizontalValue = Color.Lerp(bottomLeftMaterial.GetColor(nameId), bottomRightMaterial.GetColor(nameId), hs);
							materialInstance.SetColor(nameId, horizontalValue);
							break;
						}
					}

					nameId = materialInstance.shader.GetPropertyNameId(verticalBlendParameter);
					switch (materialInstance.shader.GetPropertyType(verticalBlendParameter))
					{
						case ShaderPropertyType.Float:
						case ShaderPropertyType.Range:
						{
							float verticalValue = lerp(bottomLeftMaterial.GetFloat(nameId), topLeftMaterial.GetFloat(nameId), hs);
							materialInstance.SetFloat(nameId, verticalValue);
							break;
						}

						case ShaderPropertyType.Color:
						{
							Color verticalValue = Color.Lerp(bottomLeftMaterial.GetColor(nameId), topLeftMaterial.GetColor(nameId), hs);
							materialInstance.SetColor(nameId, verticalValue);
							break;
						}
					}
				}

				instance.material = materialInstance;
			}
		}
	}
}