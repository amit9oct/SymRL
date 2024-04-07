from kivy.app import App
from kivy.uix.textinput import TextInput
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
import traceback

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
        if 'policy' in kwargs:
            self.policy = kwargs['policy']
            del kwargs['policy']
        else:
            self.policy = None
        if 'maximum_step_limit' in kwargs:
            self.maximum_step_limit = kwargs['maximum_step_limit']
            del kwargs['maximum_step_limit']
        else:
            self.maximum_step_limit = 20
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

        self.buttons_grid = GridLayout(cols=8, size_hint_y=None, height=100)
        self.add_widget(self.buttons_grid)
        self.actions = self.env.action_space.actions
        self.action_index_map = {action: i for i, action in enumerate(self.actions)}
        self.create_action_buttons()


        self.equation_text_input = TextInput(text=self.equation, size_hint_y=None, height=50)
        self.add_widget(self.equation_text_input)

        self.stop_event = Event()
        
        self.last_buttons_grid = GridLayout(cols=3, size_hint_y=None, height=50)
        self.add_widget(self.last_buttons_grid)
        self.reset_button = Button(text='Reset', size_hint_y=None, height=50)
        self.reset_button.bind(on_press=self.on_reset)
        self.last_buttons_grid.add_widget(self.reset_button)
        self.set_eqn_button = Button(text='Set Equation', size_hint_y=None, height=50)
        self.set_eqn_button.bind(on_press=self.on_equation_entered)
        self.last_buttons_grid.add_widget(self.set_eqn_button)
        self.run_policy_button = Button(text='Run Policy', size_hint_y=None, height=50)
        self.run_policy_button.bind(on_press=self.on_policy_run)
        self.last_buttons_grid.add_widget(self.run_policy_button)
        self.on_start()
    
    def on_policy_run(self, instance):
        if self.policy is not None:
            done = False
            step_cnt = 0
            state = self.env.reset()
            while not done and step_cnt < self.maximum_step_limit:
                action = self.policy.select_action(state)
                state, done = self.take_action(action, is_index=True)
                step_cnt += 1

    def on_equation_entered(self, instance):
        # Placeholder for handling the entered equation
        equation = self.equation_text_input.text
        self.update_equation(equation)
        self.update_solution(None, reset=True)
        self.env.reset([equation])
        self.render_eqn()
    
    def on_start(self):
        # Schedule a check for the stop_event in the background
        Clock.schedule_interval(lambda dt: self.check_for_stop_event(), 1)
        Clock.schedule_interval(lambda dt: self.render_eqn(), 0.1)

    def create_action_buttons(self):
        for action in self.actions:
            btn = Button(text=action, size_hint_y=None, height=50)
            btn.bind(on_press=self.on_action)
            self.buttons_grid.add_widget(btn)
    
    def check_for_stop_event(self):
        if self.stop_event.is_set():
            # If the event is set, stop the app
            App.get_running_app().stop()
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
    
    def render_eqn(self):
        self.equation_label.text = self.equation
        self.info_label.text = self.info_text
        self.solution_scroll.update_text(self.solution_text)
    
    def on_action(self, instance):
        # Placeholder for handling the entered action
        action = instance.text
        self.take_action(action)
        self.render_eqn()
        # Here you can call env.step(action) or similar
    
    def take_action(self, action, is_index=False):
        # Placeholder for taking action
        action_index = self.action_index_map.get(action, -1) if not is_index else action
        if action_index == action:
            action = self.actions[action_index]
        try:
            next_state, _, done, _, _ = self.env.step(action_index)
        except Exception as e:
            print(f"Error in taking action: {e}")
            print(f"stack trace: {traceback.format_exc()}")
            raise
        self.update_solution(f"{next_state} [{action}] [done={done}]")
        return next_state, done
        
    def on_reset(self, instance):
        # Placeholder for the reset action
        self.equation = str(self.env.reset())
        self.update_solution(None, reset=True)
        self.update_equation(self.equation)
        self.render_eqn()
        # Reset the environment and update the equation display
        # self.update_equation(reset_equation_from_env)

class EquationApp(App):
    def __init__(self, env: gym.Env, **kwargs):
        super(EquationApp, self).__init__()
        self.env = env
        self.gui = EquationGUI(self.env, **kwargs)
    
    def build(self):
        return self.gui

if __name__ == '__main__':
    from sympy_env import SympyEnv
    env = SympyEnv(['2*x + 4*x - 4 = 10'])
    app = EquationApp(env)
    app.run()