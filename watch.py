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


import os
import pandas as pd
import time
import glob

class PollingProcessor:
    def __init__(self, trigger_dir, data_dir, interval=10):
        self.trigger_dir = trigger_dir
        self.data_dir = data_dir
        self.interval = interval
        self.processed_triggers = set()

    def fetch_new_triggers(self):
        trigger_files = glob.glob(os.path.join(self.trigger_dir, '*.csv'))
        new_files = [f for f in trigger_files if f not in self.processed_triggers]
        return new_files

    def process_trigger(self, file_path):
        try:
            triggers_df = pd.read_csv(file_path)
            for _, trigger in triggers_df.iterrows():
                ai_req_id = trigger['AI_REQ_ID']
                version = trigger['Version']
                plant = trigger['Plant']
                task_id = trigger['Task_ID']
                user_id = trigger['User_ID']

                folder_path = os.path.join(self.data_dir, plant, task_id, user_id, 'IN')
                pattern = os.path.join(folder_path, '**', f"{ai_req_id}_{version}_*.csv")
                matching_files = glob.glob(pattern, recursive=True)

                for file in matching_files:
                    data_df = pd.read_csv(file)
                    print(f"Processed file: {file}")
                    print(data_df.head())

            self.processed_triggers.add(file_path)
        except Exception as e:
            print(f"Error processing trigger file {file_path}: {e}")

    def run(self):
        print("Starting polling processor...")
        while True:
            new_triggers = self.fetch_new_triggers()
            if new_triggers:
                print(f"Detected {len(new_triggers)} new trigger file(s).")
                for trigger_file in new_triggers:
                    self.process_trigger(trigger_file)
            time.sleep(self.interval)

if __name__ == "__main__":
    trigger_directory = "/path/to/REQ_POOL"
    data_directory = "/path/to/data"
    processor = PollingProcessor(trigger_directory, data_directory, interval=30)
    processor.run()
