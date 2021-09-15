using System;
using JetBrains.Annotations;
using Runtime.EntityTypes;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine.Assertions;
using static Unity.Mathematics.math;

namespace Runtime
{
	enum EntityType
	{
		None,
		Sphere,
		Rect,
		Box,
		Triangle,
		Mesh
	}

	readonly unsafe struct Entity
	{
		public readonly EntityType Type;
		public readonly bool Moving;
		public readonly RigidTransform OriginTransform;
		public readonly RigidTransform InverseTransform;
		public readonly float3 DestinationOffset;
		public readonly float2 TimeRange;
		public readonly Material* Material;

		[NativeDisableUnsafePtrRestriction] public readonly void* Content;

		public Entity(EntityType type, void* contentPointer, RigidTransform originTransform, Material* material,
			bool moving = false, float3 destinationOffset = default, float2 timeRange = default) : this()
		{
			Type = type;
			Content = contentPointer;
			Moving = moving;
			TimeRange = timeRange;
			OriginTransform = originTransform;
			DestinationOffset = destinationOffset;
			TimeRange = timeRange;
			Material = material;

			if (!moving)
				InverseTransform = inverse(OriginTransform);
			else
				Assert.AreNotEqual(timeRange.x, timeRange.y, "Time range cannot be empty for moving entities.");
		}

		[Pure]
		public bool Hit(Ray ray, float tMin, float tMax, Material* randomWalkEntryDensity, ref RandomSource rng, out HitRecord rec)
		{
			if (HitInternal(ray, tMin, tMax, randomWalkEntryDensity, ref rng,
				out float distance, out float3 entityLocalNormal, out RigidTransform transformAtTime, out _))
			{
				// TODO: normal is disregarded for isotropic materials
				rec = new HitRecord(distance, ray.GetPoint(distance), normalize(rotate(transformAtTime, entityLocalNormal)));
				return true;
			}

			rec = default;
			return false;
		}

		bool HitInternal(Ray ray, float tMin, float tMax, Material* randomWalkEntryMaterial, ref RandomSource rng,
			out float distance, out float3 entitySpaceNormal, out RigidTransform transformAtTime, out Ray entitySpaceRay)
		{
			RigidTransform inverseTransform;

			if (!Moving)
			{
				transformAtTime = OriginTransform;
				inverseTransform = InverseTransform;
			}
			else
			{
				transformAtTime = TransformAtTime(ray.Time);
				inverseTransform = inverse(transformAtTime);
			}

			entitySpaceRay = new Ray(
				transform(inverseTransform, ray.Origin),
				rotate(inverseTransform, ray.Direction));

			if (!HitContent(entitySpaceRay, tMin, tMax, out distance, out entitySpaceNormal))
				return false;

			if (randomWalkEntryMaterial != null)
			{
				float volumeHitDistance = -(1 / randomWalkEntryMaterial->Density) * log(rng.NextFloat());

				if (volumeHitDistance < distance)
					distance = volumeHitDistance;
				else
				{
					// Accept the hit only if it's not a probablistic volume exit hit
					return Material->Type != MaterialType.ProbabilisticVolume;
				}
			}

			return true;
		}

		bool HitContent(Ray r, float tMin, float tMax, out float distance, out float3 normal)
		{
			switch (Type)
			{
				case EntityType.Sphere: return ((Sphere*) Content)->Hit(r, tMin, tMax, out distance, out normal);
				case EntityType.Rect: return ((Rect*) Content)->Hit(r, tMin, tMax, out distance, out normal);
				case EntityType.Box: return ((Box*) Content)->Hit(r, tMin, tMax, out distance, out normal);
				case EntityType.Triangle: return ((Triangle*) Content)->Hit(r, tMin, tMax, out distance, out normal);

				default:
					distance = 0;
					normal = default;
					return false;
			}
		}

		public float Pdf(Ray r, ref RandomSource rng)
		{
			if (HitInternal(r, 0.001f, float.PositiveInfinity, null, ref rng,
				out float distance, out float3 entitySpaceNormal, out _, out Ray entitySpaceRay))
			{
				switch (Type)
				{
					case EntityType.Rect:
						return ((Rect*) Content)->Pdf(entitySpaceRay.Direction, distance, entitySpaceNormal);

					case EntityType.Sphere:
						return ((Sphere*) Content)->Pdf(entitySpaceRay.Origin);

					default: throw new NotImplementedException();
				}
			}

			return 0;
		}

		public float3 RandomPoint(float time, ref RandomSource rng)
		{
			float3 localPoint;
			switch (Type)
			{
				case EntityType.Rect: localPoint = ((Rect*) Content)->RandomPoint(ref rng); break;
				case EntityType.Sphere: localPoint = ((Sphere*) Content)->RandomPoint(ref rng); break;
				default: throw new NotImplementedException();
			}

			return transform(TransformAtTime(time), localPoint);
		}

		public RigidTransform TransformAtTime(float t) =>
			new RigidTransform(OriginTransform.rot,
				OriginTransform.pos +
				DestinationOffset * clamp(unlerp(TimeRange.x, TimeRange.y, t), 0.0f, 1.0f));
	}
}