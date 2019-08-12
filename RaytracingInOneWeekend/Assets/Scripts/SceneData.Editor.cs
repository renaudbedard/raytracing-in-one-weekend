using Sirenix.OdinInspector.Editor;
using Sirenix.Utilities.Editor;
using UnityEditor;
using UnityEngine;
using static Unity.Mathematics.math;

namespace RaytracerInOneWeekend
{
	[CustomEditor(typeof(Raytracer))]
	public class RaytracerEditor : OdinEditor
	{
		bool mouseButtonWasDown;
		EntityData hotEntity;

		protected override void OnEnable()
		{
			Tools.hidden = true;
		}

		protected override void OnDisable()
		{
			Tools.hidden = false;
		}

		void OnSceneGUI()
		{
			// to allow for mouse-up event detection
			// see : https://forum.unity.com/threads/hurr-eventtype-mouseup-not-working-on-left-clicks.99909/#post-3594727
			HandleUtility.AddDefaultControl(GUIUtility.GetControlID(FocusType.Passive));

			var raytracer = (Raytracer) target;
			Event currentEvent = Event.current;

			if (currentEvent.OnMouseDown(0, false))
				mouseButtonWasDown = true;

			if (currentEvent.OnMouseMoveDrag(false))
				mouseButtonWasDown = false;

			if (mouseButtonWasDown && currentEvent.OnMouseUp(0, false))
			{
				mouseButtonWasDown = false;
				UnityEngine.Ray viewRay = HandleUtility.GUIPointToWorldRay(Event.current.mousePosition);

				EntityData lastHotEntity = hotEntity;
				if (lastHotEntity != null)
				{
					hotEntity.Selected = false;
					hotEntity = null;
				}

				float minDistance = float.MaxValue;
				foreach (EntityData entity in raytracer.activeEntities)
				{
					float distance = HandleUtility.DistancePointLine(entity.Center, viewRay.origin, viewRay.direction * 1000);

					if (new Bounds(entity.Center, entity.Size).IntersectRay(viewRay) &&
					    distance < minDistance)
					{
						hotEntity = entity;
						minDistance = distance;
					}
				}

				if (hotEntity != null) hotEntity.Selected = true;
				if (hotEntity != lastHotEntity) HandleUtility.Repaint();
			}

			if (hotEntity == null) return;

			if (currentEvent.OnKeyDown(KeyCode.Delete))
			{
				Undo.RecordObject(raytracer.scene, "Disabled object");
				hotEntity.Enabled = false;
				hotEntity.Selected = false;
				hotEntity.MarkDirty();
				hotEntity = null;
				return;
			}

			Vector3 currentPosition = hotEntity.Center;

			switch (Tools.current)
			{
				case Tool.Move:
					EditorGUI.BeginChangeCheck();
					Vector3 newPosition = Handles.PositionHandle(currentPosition, Quaternion.identity);
					if (EditorGUI.EndChangeCheck())
					{
						Undo.RecordObject(raytracer.scene, "Moved object");

						switch (hotEntity.Type)
						{
							case EntityType.Sphere:
								hotEntity.SphereData.Center = newPosition;
								break;

							case EntityType.Rect:
								hotEntity.RectData.Center = newPosition;
								hotEntity.RectData.Distance = newPosition.z;
								break;
						}

						hotEntity.MarkDirty();
					}
					break;

				case Tool.Scale:
					Vector3 currentScale = hotEntity.Size;
					float size = HandleUtility.GetHandleSize(currentPosition);

					EditorGUI.BeginChangeCheck();
					Vector3 newSize = Handles.ScaleHandle(currentScale, currentPosition, Quaternion.identity, size);
					if (EditorGUI.EndChangeCheck())
					{
						Undo.RecordObject(raytracer.scene, "Scaled object");

						switch (hotEntity.Type)
						{
							case EntityType.Sphere:
								hotEntity.SphereData.Radius = dot(newSize, 1) / 3 / 2;
								break;

							case EntityType.Rect:
								hotEntity.RectData.Size = newSize;
								break;
						}

						hotEntity.MarkDirty();
					}
					break;
			}
		}
	}
}