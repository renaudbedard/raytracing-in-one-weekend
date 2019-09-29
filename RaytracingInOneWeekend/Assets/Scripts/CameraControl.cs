using UnityEngine;
using UnityEngine.InputSystem;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
#else
using OdinMock;
#endif

namespace RaytracerInOneWeekend
{
	[RequireComponent(typeof(Camera))]
	class CameraControl : MonoBehaviour
	{
		[ReadOnly] [SerializeField] new UnityEngine.Camera camera = null;
		[SerializeField] Raytracer raytracer = null;

		[SerializeField] [Range(0, 100)] float movementSpeed = 1;

		Vector3 orbitCenter;
		float dragDistance;
		bool freeLook;

		void Reset()
		{
			camera = GetComponent<UnityEngine.Camera>();
		}

		void Update()
		{
			var mouse = Mouse.current;
			var keyboard = Keyboard.current;
			var dt = Time.deltaTime;

			float speed = keyboard.leftShiftKey.isPressed ? 5 : 1;
			speed *= movementSpeed;

			if (keyboard.wKey.isPressed) transform.Translate(dt * speed * Vector3.forward, Space.Self);
			if (keyboard.sKey.isPressed) transform.Translate(dt * speed * Vector3.back, Space.Self);
			if (keyboard.aKey.isPressed) transform.Translate(dt * speed * Vector3.left, Space.Self);
			if (keyboard.dKey.isPressed) transform.Translate(dt * speed * Vector3.right, Space.Self);

			if (keyboard.qKey.isPressed) transform.Translate(dt * speed * Vector3.up, Space.Self);
			if (keyboard.eKey.isPressed) transform.Translate(dt * speed * Vector3.down, Space.Self);

			var mouseDelta = mouse.delta.ReadValue();

			var scrollValue = mouse.scroll.ReadValue();
			if (!Mathf.Approximately(scrollValue.y, 0))
				transform.Translate(scrollValue.y / 360 * Vector3.forward, Space.Self);

			if (keyboard.escapeKey.wasPressedThisFrame)
				freeLook = !freeLook;

			if (freeLook || mouse.leftButton.isPressed)
			{
				if (mouseDelta != Vector2.zero)
				{
					var mouseMovement = mouseDelta;
					mouseMovement /= Screen.dpi;
					mouseMovement *= 5;

					transform.Rotate(-mouseMovement.y, mouseMovement.x, 0, Space.Self);

					ResetRoll();
				}
			}
			else if (mouse.rightButton.isPressed)
			{
				if (mouse.rightButton.wasPressedThisFrame)
				{
					var origin = transform.localPosition;
					var forward = transform.forward;
					float distance = 1;
					if (raytracer.HitWorld(new Ray(origin, forward), out HitRecord hitRec))
					{
						distance = hitRec.Distance;
					}
					orbitCenter = origin + forward * distance;
				}

				if (mouseDelta != Vector2.zero)
				{
					var mouseMovement = mouseDelta;
					mouseMovement /= Screen.dpi;
					mouseMovement *= 10;

					transform.RotateAround(orbitCenter, transform.right, -mouseMovement.y);
					transform.RotateAround(orbitCenter, transform.up, mouseMovement.x);

					ResetRoll();
				}
			}
			else if (mouse.middleButton.isPressed)
			{
				if (mouse.middleButton.wasPressedThisFrame)
				{
					dragDistance = 1;

					var origin = transform.localPosition;
					var forward = transform.forward;
					if (raytracer.HitWorld(new Ray(origin, forward), out HitRecord hitRec))
					{
						dragDistance = hitRec.Distance;
					}
				}

				if (mouseDelta != Vector2.zero)
				{
					var mouseMovement = mouseDelta;
					mouseMovement /= Screen.dpi * 15;
					mouseMovement *= dragDistance;

					transform.Translate(-mouseMovement.x, -mouseMovement.y, 0, Space.Self);
				}
			}
		}

		void ResetRoll()
		{
			var eulers = transform.eulerAngles;
			eulers.z = 0;
			transform.eulerAngles = eulers;
		}
	}
}