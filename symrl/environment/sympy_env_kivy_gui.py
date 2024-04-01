from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from threading import Event
import gymnasium as gym
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView
from kivy.graphics import Color, Line
from kivy.clock import Clock
from kivy.properties import NumericProperty

class ScrollableLabel(ScrollView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label = Label(size_hint_y=None, markup=True)
        self.add_widget(self.label)

        # Ensure the label text wraps at the width of the ScrollView
        self.label.bind(width=lambda *x: self._update_text_width())
        self.bind(width=lambda *x: self._update_text_width())

        with self.canvas.before:
            Color(1, 0, 0, 1)  # Set the border color to red
            self.border = Line(rectangle=(self.x + 1, self.y + 1, self.width - 2, self.height - 2), width=2)
        self.bind(pos=self._update_border, size=self._update_border)

    def update_text(self, new_text):
        self.label.text = new_text
        # Update the label height to fit the new text content
        self.label.texture_update()
        self.label.height = self.label.texture_size[1]
        self.scroll_y = 1  # Scroll to the top

    def _update_text_width(self):
        # Set the label's text_size width to match the ScrollView's width
        self.label.text_size = (self.width - 20, None)  # Subtract some pixels to account for padding/border

    def _update_border(self, *args):
        self.border.rectangle = (self.x + 1, self.y + 1, self.width - 2, self.height - 2)

class EquationGUI(BoxLayout):
    text_width = NumericProperty(Window.size[0] - 100)  # Assuming a margin
    def __init__(self, env: gym.Env, **kwargs):
        super(EquationGUI, self).__init__(orientation='vertical', **kwargs)
        
        # Assuming 'env' is your environment instance with an 'equation' attribute
        self.env = env
        self.equation = str(self.env.reset())
        self.solution_text = "Solution:"
        self.info_text = "Other Information:"

        # Bind text_width to update dynamically with window size
        Window.bind(size=self._update_text_width)


        self.equation_label = Label(text=self.equation,
                                    size_hint_y=None,
                                    height=50,
                                    text_size=(self.text_width, None),
                                    valign='top')
        self.add_widget(self.equation_label)

        self.info_label = Label(text=self.info_text,
                    size_hint_y=None, 
                    height=100,
                    text_size=(self.text_width, None),
                    valign='top')
        self.add_widget(self.info_label)

        # Solution label within a ScrollView
        self.solution_scroll = ScrollableLabel(
                size_hint=(1, None), 
                height=200)
        self.add_widget(self.solution_scroll)
        self.solution_scroll.update_text(self.solution_text)

        self.buttons_grid = GridLayout(cols=6, size_hint_y=None, height=200)
        self.add_widget(self.buttons_grid)
        self.actions = self.env.action_space.actions
        self.action_index_map = {action: i for i, action in enumerate(self.actions)}
        self.create_action_buttons()

        self.reset_button = Button(text='Reset', size_hint_y=None, height=50)
        self.reset_button.bind(on_press=self.on_reset)
        self.add_widget(self.reset_button)
        self.stop_event = Event()
    
    def on_start(self):
        # Schedule a check for the stop_event in the background
        Clock.schedule_interval(self.check_for_stop_event, 10)
        Clock.schedule_interval(self.render, 1/60)

    def create_action_buttons(self):
        for action in self.actions:
            btn = Button(text=action, size_hint_y=None, height=50)
            btn.bind(on_press=self.on_action)
            self.buttons_grid.add_widget(btn)
    
    def check_for_stop_event(self):
        if self.stop_event.is_set():
            # If the event is set, stop the app
            self.stop()
            return False  # To unschedule the callback

    def _update_text_width(self, instance, value):
        # Adjust text width based on window size with some margin
        self.text_width = value[0] - 100  # Update text width based on window width

    def set_stop(self):
        # Stop the app
        self.stop_event.set()

    def update_equation(self, equation):
        # This method is safe to call from any thread
        self.equation = equation
    
    def update_solution(self, solution, reset=False):
        # This method is safe to call from any thread
        if reset:
            self.solution_text = "Solution:"
        else:
            self.solution_text += f'\n=> {solution}'
    
    def update_info(self, info):
        # This method is safe to call from any thread
        self.info_text = f"Other Information:\n{info}"
    
    def render(self):
        self.equation_label.text = self.equation
        self.info_label.text = self.info_text
        self.solution_scroll.update_text(self.solution_text)
    
    def on_action(self, instance):
        # Placeholder for handling the entered action
        action = instance.text
        action_index = self.action_index_map.get(action, -1)
        next_state, _, done, _, _ = self.env.step(action_index)
        self.update_solution(f"{next_state} [{action}] [done={done}]")
        self.render()
        # Here you can call env.step(action) or similar
        
    def on_reset(self, instance):
        # Placeholder for the reset action
        self.equation = str(self.env.reset())
        self.update_solution(None, reset=True)
        self.update_equation(self.equation)
        self.render()
        # Reset the environment and update the equation display
        # self.update_equation(reset_equation_from_env)

class EquationApp(App):
    def __init__(self, env: gym.Env):
        super(EquationApp, self).__init__()
        self.env = env
        self.gui = EquationGUI(self.env)
    
    def build(self):
        return self.gui

if __name__ == '__main__':
    from sympy_env import SympyEnv
    env = SympyEnv(['2*x + 4*x - 4 = 10'])
    app = EquationApp(env)
    app.run()