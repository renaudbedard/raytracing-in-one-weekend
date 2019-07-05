using UnityEngine;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
#endif

namespace RaytracerInOneWeekend
{
    [CreateAssetMenu]
    class MaterialData : ScriptableObject
    {
        [SerializeField] MaterialType type = MaterialType.None;
        [SerializeField] Color albedo = Color.white;
        
#if ODIN_INSPECTOR        
        [ShowIf(nameof(Type), MaterialType.Metal)]
#endif
        [Range(0, 1)] [SerializeField] float fuzz = 0;
        
#if ODIN_INSPECTOR        
        [ShowIf(nameof(Type), MaterialType.Dielectric)]
#endif
        [Range(1, 2.65f)] [SerializeField] float refractiveIndex = 1;

        public MaterialType Type => type;
        public Color Albedo => albedo;
        public float Fuzz => fuzz;
        public float RefractiveIndex => refractiveIndex;
        
#if UNITY_EDITOR
        public bool Dirty { get; set; }

        void OnValidate()
        {
            Dirty = true;
        }
#endif
    }
}