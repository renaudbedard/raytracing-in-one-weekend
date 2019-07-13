using System;
using UnityEngine;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
#endif

namespace RaytracerInOneWeekend
{
    [Serializable]
    class SphereData
    {
        [SerializeField] bool enabled = true;
        [SerializeField] Vector3 center = Vector3.zero;
        [SerializeField] float radius = 1;
        
        [SerializeField] 
#if ODIN_INSPECTOR        
        [InlineEditor] 
#endif
        MaterialData material = null;
        
        public SphereData(Vector3 center, float radius, MaterialData material)
        {
            this.center = center;
            this.radius = radius;
            this.material = material;
        }        
        
        public bool Enabled => enabled;
        public Vector3 Center => center;
        public float Radius => radius;
        public MaterialData Material => material;
    }
}