using Runtime.EntityTypes;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using Util;
using static Unity.Mathematics.math;

namespace Runtime.Jobs
{
	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	unsafe struct AddMeshRuntimeEntitiesJob : IJob
	{
		[ReadOnly] public Mesh.MeshDataArray MeshDataArray;
		[ReadOnly] [NativeDisableUnsafePtrRestriction] public Material* Material;

		public NativeList<Triangle> Triangles;
		public NativeList<Entity> Entities;

		public bool FaceNormals, Moving;
		public RigidTransform RigidTransform;
		public float3 DestinationOffset;
		public float2 TimeRange;
		public float Scale;

		public void Execute()
		{
			float3* worldSpaceVertices = stackalloc float3[3];
			float3* worldSpaceNormals = stackalloc float3[3];
			float2* triangleUv = stackalloc float2[3];

			for (int meshIndex = 0; meshIndex < MeshDataArray.Length; meshIndex++)
			{
				Mesh.MeshData meshData = MeshDataArray[meshIndex];

				using var vertices = new NativeArray<Vector3>(meshData.vertexCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
				meshData.GetVertices(vertices);

				NativeArray<Vector3> normals = default;
				if (!FaceNormals)
				{
					normals = new NativeArray<Vector3>(meshData.vertexCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
					meshData.GetNormals(normals);
				}

				NativeArray<Vector2> texCoords = default;
				if (meshData.HasVertexAttribute(VertexAttribute.TexCoord0))
				{
					texCoords = new NativeArray<Vector2>(meshData.vertexCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
					meshData.GetUVs(0, texCoords);
				}

				NativeArray<ushort> indices = meshData.GetIndexData<ushort>();
				for (int i = 0; i < indices.Length; i += 3)
				{
					int3 triangleIndices = int3(indices[i], indices[i + 1], indices[i + 2]);

					// Bake transform
					for (int j = 0; j < 3; j++)
					{
						worldSpaceVertices[j] = transform(RigidTransform, vertices[triangleIndices[j]] * Scale);
						if (!FaceNormals) worldSpaceNormals[j] = mul(RigidTransform.rot, normals[triangleIndices[j]]);
						triangleUv[j] = texCoords.IsCreated ? texCoords[triangleIndices[j]] : default;
					}

					if (FaceNormals)
						Triangles.AddNoResize(new Triangle(
							worldSpaceVertices[0], worldSpaceVertices[1], worldSpaceVertices[2],
							triangleUv[0], triangleUv[1], triangleUv[2]));
					else
						Triangles.AddNoResize(new Triangle(
							worldSpaceVertices[0], worldSpaceVertices[1], worldSpaceVertices[2],
							worldSpaceNormals[0], worldSpaceNormals[1], worldSpaceNormals[2],
							triangleUv[0], triangleUv[1], triangleUv[2]));

					var contentPointer = Triangles.GetUnsafeList()->Ptr + (Triangles.Length - 1);

					Entity entity = Moving
						? new Entity(EntityType.Triangle, contentPointer, default, Material, true, DestinationOffset, TimeRange)
						: new Entity(EntityType.Triangle, contentPointer, default, Material);

					Entities.AddNoResize(entity);
				}

				normals.SafeDispose();
				texCoords.SafeDispose();
			}
		}
	}
}