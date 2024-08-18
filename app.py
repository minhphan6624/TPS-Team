from tkinter import *

class App:
    def __init__(self, graph):
        self.graph = graph
        # self.path = path
        self.start_entry = ""
        self.end_entry = ""

        # create the window
        self.window = Tk()
        self.window.title("TrafficPredictionSystem")

        # create fields
        # start input
        Label(self.window, text="Start SCAT Number:").grid(row=0, column=0, padx=10, pady=10)
        self.start_entry = Entry(self.window)
        self.start_entry.grid(row=0, column=1, padx=10, pady=10)
        # end input
        Label(self.window, text="End SCAT Number:").grid(row=1, column=0, padx=10, pady=10)
        self.end_entry = Entry(self.window)
        self.end_entry.grid(row=1, column=1, padx=10, pady=10)
        # submit button
        submit_button = Button(self.window, text="Submit", command=self.submit)
        submit_button.grid(row=2, column=0, columnspan=2, pady=20)
       
        
    def run(self):
        # run app
        self.window.mainloop()

    
    def get_scat_numbers(self):
        start = self.start_entry.get()
        end = self.end_entry.get()
        return start, end


    def submit(self):
        start_scat, end_scat = self.get_scat_numbers()
        print(f"Start SCAT Number: {start_scat}")
        print(f"End SCAT Number: {end_scat}")
        self.window.quit()
        