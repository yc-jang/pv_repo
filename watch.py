import os
import glob
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class TriggerHandler(FileSystemEventHandler):
    def __init__(self, base_dir, data_dir):
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.processed_triggers = set()

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.csv'):
            return
        print(f"New trigger file detected: {event.src_path}")
        self.handle_trigger(event.src_path)

    def handle_trigger(self, trigger_file):
        try:
            triggers_df = pd.read_csv(trigger_file)
            for _, trigger in triggers_df.iterrows():
                ai_req_id = trigger['AI_REQ_ID']
                version = trigger['Version']
                plant = trigger['Plant']
                task_id = trigger['Task_ID']
                user_id = trigger['User_ID']

                trigger_id = (ai_req_id, version, plant, task_id, user_id)
                if trigger_id in self.processed_triggers:
                    continue

                self.processed_triggers.add(trigger_id)

                folder_path = os.path.join(self.data_dir, plant, task_id, user_id, 'IN')
                pattern = os.path.join(folder_path, '**', f"{ai_req_id}_{version}_*.csv")
                matching_files = glob.glob(pattern, recursive=True)

                for file_path in matching_files:
                    df = pd.read_csv(file_path)
                    print(f"Processed file: {file_path}")
                    print(df.head())

        except Exception as e:
            print(f"Error processing trigger file {trigger_file}: {e}")

def monitor_triggers(trigger_dir, data_dir):
    event_handler = TriggerHandler(trigger_dir, data_dir)
    observer = Observer()
    observer.schedule(event_handler, path=trigger_dir, recursive=False)
    observer.start()
    print(f"Monitoring trigger directory: {trigger_dir}")

    try:
        while True:
            pass
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

if __name__ == "__main__":
    trigger_directory = "/path/to/REQ_POOL"
    data_directory = "/path/to/data"
    monitor_triggers(trigger_directory, data_directory)
