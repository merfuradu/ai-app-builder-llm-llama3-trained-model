import customtkinter
import customtkinter as ctk
import tkinter
from ai_trained_model import generate_proposal  # Import the generate_proposal function from main_logic.py
import threading

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")
# Initialize the main window
app = ctk.CTk()

# Set the title and size of the window
app.title("AI Proposal Generator")
app.geometry("500x400")

loading_label = ctk.CTkLabel(app, text="Loading...", text_color="red", font=("Arial", 14))
loading_label.pack_forget()
def set_loading(is_loading):
    if is_loading:
        loading_label.pack(pady=5)
    else:
        loading_label.pack_forget()

def update_output(proposal):
    # Clear the existing output
    for widget in output_frame.winfo_children():
        widget.destroy()

    output_label = ctk.CTkLabel(output_frame, text=proposal, wraplength=550, anchor="w", justify="left")
    output_label.pack(pady=5, padx=5)

# Define a function to display the generated proposal
def display_text():
    def run_generate_proposal():
        user_text = entry.get()  # Get the text input from the user
        set_loading(True)  # Show loading indicator
        proposal = generate_proposal(user_text)  # Call the generate_proposal function with the user input
        set_loading(False)  # Hide loading indicator
        update_output(proposal)  # Display the result in the output label

    # Run the proposal generation in a separate thread to avoid freezing the UI
    threading.Thread(target=run_generate_proposal).start()

# label
label = ctk.CTkLabel(app, text="Enter project details:", font=("Arial", 16))
label.pack(pady=10)

#entry widget
entry = ctk.CTkEntry(app, width=300)
entry.pack(pady=10)

#button to submit
button = ctk.CTkButton(app, text="Generate Proposal", command=display_text)
button.pack(pady=10)

#output label
output_label_title = ctk.CTkLabel(app, text="The resulted output:", anchor="w", justify="left")
output_label_title.pack(pady=10, padx=10, anchor="w")

#label to display the output
output_frame = ctk.CTkScrollableFrame(app, width=550, height=300)
output_frame.pack(pady=10, padx=10)

# Run the application
app.mainloop()
