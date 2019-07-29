#if !ODIN_INSPECTOR
using System;

namespace OdinMock
{
	public class ShowIfAttribute : Attribute
	{
		public ShowIfAttribute(string _, object __) { }
		public ShowIfAttribute(string _) { }
	}

	public class HideIfAttribute : Attribute
	{
		public HideIfAttribute(string _, object __) { }
	}

	public class MinMaxSliderAttribute : Attribute
	{
		public MinMaxSliderAttribute(float _, float __, bool ___) { }
	}

	public class LabelWidthAttribute : Attribute
	{
		public LabelWidthAttribute(float _) { }
	}

	public class InlineButtonAttribute : Attribute
	{
		public InlineButtonAttribute(string _, string __) { }
	}

	public class InlineEditorAttribute : Attribute
	{
		public bool DrawHeader;
	}

	public class ShowInInspectorAttribute : Attribute
	{
	}

	public class ReadOnlyAttribute : Attribute
	{
	}

	public class ValueDropdownAttribute : Attribute
	{
		public ValueDropdownAttribute(string _) { }
	}

	public class DisableInPlayModeAttribute : Attribute
	{
	}

	public class DisableInEditorModeAttribute : Attribute
	{
	}

	public class ButtonAttribute : Attribute
	{
	}

	public class DisableIfAttribute : Attribute
	{
		public DisableIfAttribute(string _) { }
	}

	public class EnableIfAttribute : Attribute
	{
		public EnableIfAttribute(string _) { }
	}

	public class ButtonGroupAttribute : Attribute
	{
		public ButtonGroupAttribute(string _) { }
	}

	public class TitleAttribute : Attribute
	{
		public TitleAttribute(string _) { }
	}
}
#endif