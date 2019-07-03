using System;
using Sirenix.OdinInspector;
using UnityEngine;

namespace RaytracerInOneWeekend
{
    [Serializable]
    class SphereData
    {
        [SerializeField] bool enabled = true;
        [SerializeField] Vector3 center = Vector3.zero;
        [SerializeField] float radius = 1;
        [SerializeField] [InlineEditor] MaterialData material = null;
        
        public bool Enabled => enabled;
        public Vector3 Center => center;
        public float Radius => radius;
        public MaterialData Material => material;
    }
}