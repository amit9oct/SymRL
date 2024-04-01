import tkinter as tk
from tkinter import ttk
from threading import Event

class SympyEnvGUI(tk.Tk):
    def __init__(self, env):
        super().__init__()
        self.env = env  # The RL environment
        self.initialize_gui()
        self.latest_equation = [str(env.equation)]
        self.context = None
        self.exit_event = Event()
        self.check_exit()
        # self.update_interval = 200  # Update interval in milliseconds
        # self.schedule_update()

    def initialize_gui(self):
        self.title('Equation Manipulation Environment')
        self.geometry('800x800')

        # Equation display
        self.context_var = tk.StringVar()
        self.context_var.set("Other Information:\n")
        self.context_label = ttk.Label(self, textvariable=self.context_var, font=('Courier', 12), wraplength=600)
        self.context_label.pack(pady=20)
        self.equation_var = tk.StringVar()
        self.equation_var.set(str(self.env.equation))
        self.equation_label = ttk.Label(self, textvariable=self.equation_var, font=('Courier', 12))
        self.equation_label.pack(pady=10)

        # Action input (for simplicity, using direct entry of action strings)
        self.action_entry = ttk.Entry(self)
        self.action_entry.pack()

        # Step button
        self.step_button = ttk.Button(self, text='Step', command=self.step_action)
        self.step_button.pack(pady=5)

        # Reset button
        self.reset_button = ttk.Button(self, text='Reset', command=self.reset_environment)
        self.reset_button.pack(pady=5)

    def check_exit(self):
        if self.exit_event.is_set():
            self.destroy()
            print("GUI destroyed")
        else:
            self.after(100, self.check_exit)

    def step_action(self):
        action = self.action_entry.get()
        try:
            observation, reward, done, _ = self.env.step(action)
            self.equation_var.set(str(observation))
            if done:
                tk.messagebox.showinfo("Solved", "Equation solved!")
        except ValueError as e:
            tk.messagebox.showerror("Error", str(e))

    def reset_environment(self):
        observation = self.env.reset()
        self.equation_var.set(str(observation))        
    
    def reset_gui(self):
        self.latest_equation = [str(self.env.equation)]
        self.update_gui_with_latest_equation()

    def update_gui_with_latest_equation(self):
        # This method updates the GUI with the latest equation
        if self.context is not None:
            self.context_var.set(f"Other Information:\n {self.context}")
        latest_equation = '\n=> '.join(self.latest_equation) if len(self.latest_equation) > 1 else self.latest_equation[0]
        self.equation_var.set(latest_equation)

    def schedule_update(self):
        # Schedule the next update
        self.update_gui_with_latest_equation()
        self.after(self.update_interval, self.schedule_update)
    
    def signal_exit(self):
        self.exit_event.set()
        print("Exit event set")

# Assuming 'env' is an instance of SympyEnv
# env = SympyEnv(equation_str="your_equation_here")
# app = SympyEnvGUI(env)
# app.mainloop()
