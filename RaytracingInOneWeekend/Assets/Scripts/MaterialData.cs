using Sirenix.OdinInspector;
using UnityEngine;

namespace RaytracerInOneWeekend
{
    [CreateAssetMenu]
    class MaterialData : ScriptableObject
    {
        [SerializeField] MaterialType type = MaterialType.None;
        [SerializeField] Color albedo = Color.white;
        [ShowIf(nameof(Type), MaterialType.Metal)] [Range(0, 1)] [SerializeField] float fuzz = 0;
        [ShowIf(nameof(Type), MaterialType.Dielectric)] [Range(1, 2.65f)] [SerializeField] float refractiveIndex = 1;

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