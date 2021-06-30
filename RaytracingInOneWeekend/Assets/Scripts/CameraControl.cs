using UnityEngine;
using UnityEngine.InputSystem;

#if ODIN_INSPECTOR
using Sirenix.OdinInspector;
#else
using OdinMock;
#endif

namespace RaytracerInOneWeekend
{
	[RequireComponent(typeof(UnityEngine.Camera))]
	class CameraControl : MonoBehaviour
	{
		[ReadOnly] [SerializeField] new UnityEngine.Camera camera = null;

		[SerializeField] [Range(0, 100)] float movementSpeed = 1;

		Vector3 orbitCenter = default;
		float dragDistance = 0;
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

			if (keyboard.escapeKey.wasPressedThisFrame)
				freeLook = !freeLook;

			var lmbPressed = mouse.leftButton.isPressed;
			var rmbPressed = mouse.rightButton.isPressed;
			var mmbPressed = mouse.middleButton.isPressed;

			var mouseViewportPosition = camera.ScreenToViewportPoint(mouse.position.ReadValue());
			bool mouseInViewport = mouseViewportPosition.x > 0 && mouseViewportPosition.x < 1 &&
			                       mouseViewportPosition.y > 0 && mouseViewportPosition.y < 1;

			if (!mouseInViewport)
				lmbPressed = rmbPressed = mmbPressed = false;

			if (freeLook || lmbPressed)
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
			else if (rmbPressed)
			{
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
			else if (mmbPressed)
			{
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