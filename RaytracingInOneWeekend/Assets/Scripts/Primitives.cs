using System;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
#if SOA_SPHERES
	struct SoaSpheres : IDisposable
	{
		public NativeArray<float> CenterX, CenterY, CenterZ;
		public NativeArray<float> SquaredRadius;
#if BUFFERED_MATERIALS
		public NativeArray<ushort> MaterialIndex;
#else
		public NativeArray<Material> Material;
#endif

		public int Count => CenterX.Length;

		public void Dispose()
		{
			if (CenterX.IsCreated) CenterX.Dispose();
			if (CenterY.IsCreated) CenterY.Dispose();
			if (CenterZ.IsCreated) CenterZ.Dispose();
			if (SquaredRadius.IsCreated) SquaredRadius.Dispose();
#if BUFFERED_MATERIALS			
			if (MaterialIndex.IsCreated) MaterialIndex.Dispose();
#else
			if (Material.IsCreated) Material.Dispose();
#endif
		}
	}
#else
	enum PrimitiveType
	{
		None,
		Sphere
	}

	struct Primitive
	{
		public readonly PrimitiveType Type;

		[ReadOnly] readonly NativeSlice<Sphere> sphere;

		// TODO: do we need a public accessor to the underlying primitive?

		public Primitive(NativeSlice<Sphere> sphere)
		{
			UnityEngine.Assertions.Assert.IsTrue(sphere.Length == 1, "Primitive cannot be multi-valued");
			Type = PrimitiveType.Sphere;
			this.sphere = sphere;
		}

		public bool Hit(Ray r, float tMin, float tMax, out HitRecord rec)
		{
			switch (Type)
			{
				case PrimitiveType.Sphere:
					return sphere[0].Hit(r, tMin, tMax, out rec);

				default:
					rec = default;
					return false;
			}
		}
	}

	struct Sphere
	{
		public readonly float3 Center;
		public readonly float SquaredRadius;
#if BUFFERED_MATERIALS
		public readonly ushort MaterialIndex;
#else
		public readonly Material Material;
#endif

		public Sphere(float3 center, float radius, 
#if BUFFERED_MATERIALS
			ushort materialIndex)
#else
			Material material)
#endif
		{
			Center = center;
			SquaredRadius = radius * radius;
#if BUFFERED_MATERIALS			
			MaterialIndex = materialIndex;
#else
			Material = material;
#endif
		}
	}
#endif

	static class WorldExtensions
	{
#if SOA_SPHERES
		public static unsafe bool Hit(this SoaSpheres spheres, Ray r, float tMin, float tMax, out HitRecord rec)
		{
			rec = new HitRecord(tMax, 0, 0, default);

#if HIT_FOUR_WIDE
			float4 a = dot(r.Direction, r.Direction);
			
			float4* pCenterX = (float4*) spheres.CenterX.GetUnsafeReadOnlyPtr(),
				pCenterY = (float4*) spheres.CenterY.GetUnsafeReadOnlyPtr(),
				pCenterZ = (float4*) spheres.CenterZ.GetUnsafeReadOnlyPtr(),
				pSqRadius = (float4*) spheres.SquaredRadius.GetUnsafeReadOnlyPtr();

			int4 curId = int4(0, 1, 2, 3), hitId = -1;
			float4 hitT = tMax;
			bool4 mask;
			
			for (int i = 0; i < spheres.Count / 4; i++)
			{
				float4 centerX = *pCenterX,
					centerY = *pCenterY,
					centerZ = *pCenterZ,
					sqRadius = *pSqRadius;

				float4 ocX = r.Origin.x - centerX,
					ocY = r.Origin.y - centerY,
					ocZ = r.Origin.z - centerZ;

				float4 b = ocX * r.Direction.x + ocY * r.Direction.y + ocZ * r.Direction.z;
				float4 c = ocX * ocX + ocY * ocY + ocZ * ocZ - sqRadius;
				float4 discriminant = b * b - a * c;

				bool4 discriminantTest = discriminant > 0;

				if (any(discriminantTest))
				{
					float4 sqrtDiscriminant = sqrt(discriminant);
					
					float4 t0 = (-b - sqrtDiscriminant) / a;
					float4 t1 = (-b + sqrtDiscriminant) / a;

					float4 t = select(t1, t0, t0 > tMin);
					mask = discriminantTest & t > tMin & t < hitT;

					hitId = select(hitId, curId, mask);
					hitT = select(hitT, t, mask);
				}
				
				pCenterX++;
				pCenterY++;
				pCenterZ++;
				pSqRadius++;
				curId += 4;
			}

			if (all(hitId == -1))
				return false;

			float minDistance = cmin(hitT);
			mask = hitT == minDistance;
			// TODO: I guess it's not impossible that more than one sphere have the same distance?
			int closestId = dot(hitId, int4(mask));

			float3 closestCenter = float3(spheres.CenterX[closestId],
				spheres.CenterY[closestId],
				spheres.CenterZ[closestId]);
			float closestRadius = sqrt(spheres.SquaredRadius[closestId]);
			var closestMaterial = spheres.Material[closestId];

			float3 point = r.GetPoint(minDistance);
			rec = new HitRecord(minDistance, point, (point - closestCenter) / closestRadius, closestMaterial);
			return true;
#else
			bool hitAnything = false;
			for (var i = 0; i < spheres.Count; i++)
				if (spheres.Hit(i, r, tMin, rec.Distance, out HitRecord thisRec))
				{
					hitAnything = true;
					rec = thisRec;
				}

			return hitAnything;
#endif
		}
#else
		public static bool Hit(this NativeArray<Primitive> primitives, Ray r, float tMin, float tMax, out HitRecord rec)
		{
			bool hitAnything = false;
			rec = new HitRecord(tMax, 0, 0, default);

			for (var i = 0; i < primitives.Length; i++)
			{
				Primitive primitive = primitives[i];
				if (primitive.Hit(r, tMin, rec.Distance, out HitRecord thisRec))
				{
					hitAnything = true;
					rec = thisRec;
				}
			}

			return hitAnything;
		}
#endif

#if SOA_SPHERES
		static bool Hit(this SoaSpheres spheres, int sphereIndex, Ray r, float tMin, float tMax, out HitRecord rec)
#else
		public static bool Hit(this Sphere s, Ray r, float tMin, float tMax, out HitRecord rec)
#endif
		{
#if SOA_SPHERES
			float3 center = float3(
				spheres.CenterX[sphereIndex], 
				spheres.CenterY[sphereIndex],
				spheres.CenterZ[sphereIndex]);
			
			float squaredRadius = spheres.SquaredRadius[sphereIndex];
#if BUFFERED_MATERIALS
			ushort material = spheres.MaterialIndex[sphereIndex];
#else
			Material material = spheres.Material[sphereIndex];
#endif
#else
			float3 center = s.Center;
			float squaredRadius = s.SquaredRadius;
#if BUFFERED_MATERIALS
			ushort material = s.MaterialIndex;
#else
			Material material = s.Material;
#endif
#endif
			float3 oc = r.Origin - center;
			float a = dot(r.Direction, r.Direction);
			float b = dot(oc, r.Direction);
			float c = dot(oc, oc) - squaredRadius;
			float discriminant = b * b - a * c;

			if (discriminant > 0)
			{
				float sqrtDiscriminant = sqrt(discriminant);
				float t = (-b - sqrtDiscriminant) / a;
				if (t < tMax && t > tMin)
				{
					float3 point = r.GetPoint(t);
					rec = new HitRecord(t, point, (point - center) / sqrt(squaredRadius), material);
					return true;
				}

				t = (-b + sqrtDiscriminant) / a;
				if (t < tMax && t > tMin)
				{
					float3 point = r.GetPoint(t);
					rec = new HitRecord(t, point, (point - center) / sqrt(squaredRadius), material);
					return true;
				}
			}

			rec = default;
			return false;
		}
	}
}