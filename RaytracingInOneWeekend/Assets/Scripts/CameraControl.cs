using UnityEngine;
using UnityEngine.InputSystem;

namespace RaytracerInOneWeekend
{
    [RequireComponent(typeof(Camera))]
    class CameraControl : MonoBehaviour
    {
        [SerializeField] Raytracer raytracer = null;
        [SerializeField] new UnityEngine.Camera camera = null;
        
        Vector2 lastMousePosition;
        Vector3 orbitCenter;

        void Reset()
        {
            camera = GetComponent<UnityEngine.Camera>();
        }

        void Awake()
        {
            transform.LookAt(Vector3.zero);
        }
        
        void Update()
        {
            var mouse = Mouse.current;
            var keyboard = Keyboard.current;
            var dt = Time.deltaTime;

            var speed = keyboard.leftShiftKey.isPressed ? 5 : 1;
            
            if (keyboard.wKey.isPressed) transform.Translate(dt * speed * Vector3.forward, Space.Self);
            if (keyboard.sKey.isPressed) transform.Translate(dt * speed * Vector3.back, Space.Self);
            if (keyboard.aKey.isPressed) transform.Translate(dt * speed * Vector3.left, Space.Self);
            if (keyboard.dKey.isPressed) transform.Translate(dt * speed * Vector3.right, Space.Self);
            
            if (keyboard.qKey.isPressed) transform.Translate(dt * speed * Vector3.up, Space.Self);
            if (keyboard.eKey.isPressed) transform.Translate(dt * speed * Vector3.down, Space.Self);
            
            var mousePosition = mouse.position.ReadValue();
            
            var mouseViewportPosition = camera.ScreenToViewportPoint(mousePosition);
            if (mouseViewportPosition.x < 0 || mouseViewportPosition.x > 1 ||
                mouseViewportPosition.y < 0 || mouseViewportPosition.y > 1)
            {
                return;
            }
            
            if (mouse.leftButton.isPressed)
            {
                if (mousePosition != lastMousePosition)
                {
                    var mouseMovement = mousePosition - lastMousePosition;
                    mouseMovement /= Screen.dpi;
                    mouseMovement *= 5;
                    
                    transform.Rotate(mouseMovement.y, -mouseMovement.x, 0, Space.Self);

                    ResetRoll();
                    
                    lastMousePosition = mousePosition;
                }
            }
            else if (mouse.rightButton.isPressed)
            {
                if (mouse.rightButton.wasPressedThisFrame)
                {
                    var origin = transform.localPosition;
                    var forward = transform.forward;
                    float distance = 1;
                    if (raytracer.Primitives.Hit(new Ray(origin, forward), 0, float.PositiveInfinity,
                        out HitRecord hitRec))
                    {
                        distance = hitRec.Distance;
                    }
                    orbitCenter = origin + forward * distance;
                }
                
                if (mousePosition != lastMousePosition)
                {
                    var mouseMovement = mousePosition - lastMousePosition;
                    mouseMovement /= Screen.dpi;
                    mouseMovement *= 10;

                    transform.RotateAround(orbitCenter, transform.right, -mouseMovement.y);
                    transform.RotateAround(orbitCenter, transform.up, mouseMovement.x);

                    ResetRoll();

                    lastMousePosition = mousePosition;
                }
            }
            else
                lastMousePosition = mousePosition;
        }

        void ResetRoll()
        {
            var eulers = transform.eulerAngles;
            eulers.z = 0;
            transform.eulerAngles = eulers;
        }
    }
}