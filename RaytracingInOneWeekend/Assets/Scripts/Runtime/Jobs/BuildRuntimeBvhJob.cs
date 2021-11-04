using Unity;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;

namespace Runtime.Jobs
{
	[BurstCompile(FloatPrecision.Medium, FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
	struct BuildRuntimeBvhJob : IJob
	{
		[ReadOnly] public NativeList<BvhNodeData> BvhNodeDataBuffer;
		[WriteOnly] public NativeArray<BvhNode> BvhNodeBuffer;
		public int NodeCount;

		int nodeIndex;

		// Runtime BVH is inserted BACKWARDS while traversing postorder, which means the first node will be the root

		unsafe BvhNode* WalkBvh(BvhNodeData* nodeData)
		{
			BvhNode* leftNode = null, rightNode = null;

			if (!nodeData->IsLeaf)
			{
				if (nodeData->Left != null) leftNode = WalkBvh(nodeData->Left);
				if (nodeData->Right != null) rightNode = WalkBvh(nodeData->Right);
			}

			BvhNodeBuffer[nodeIndex] = new BvhNode(nodeData->Bounds, nodeData->EntitiesStart, nodeData->EntityCount,
				leftNode, rightNode);
			return (BvhNode*) BvhNodeBuffer.GetUnsafePtr() + nodeIndex--;
		}

		public unsafe void Execute()
		{
			nodeIndex = NodeCount - 1;
			WalkBvh(BvhNodeDataBuffer.GetUnsafeList()->Ptr);
		}
	}

}