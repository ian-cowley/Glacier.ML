using System.Runtime.InteropServices;

namespace Glacier.ML.Trees;

/// <summary>
/// Compact unmanaged tree node struct stored in contiguous arrays to bypass object overhead.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 4)]
public struct DecisionTreeNode
{
    public int FeatureIndex;
    public float Threshold;
    public int LeftChild;
    public int RightChild;
    public float LeafValue;
    public bool IsLeaf;

    public static DecisionTreeNode CreateLeaf(float value)
    {
        return new DecisionTreeNode
        {
            FeatureIndex = -1,
            Threshold = 0f,
            LeftChild = -1,
            RightChild = -1,
            LeafValue = value,
            IsLeaf = true
        };
    }

    public static DecisionTreeNode CreateBranch(int featureIndex, float threshold, int leftChild, int rightChild)
    {
        return new DecisionTreeNode
        {
            FeatureIndex = featureIndex,
            Threshold = threshold,
            LeftChild = leftChild,
            RightChild = rightChild,
            LeafValue = 0f,
            IsLeaf = false
        };
    }
}
