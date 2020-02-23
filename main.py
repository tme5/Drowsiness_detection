from detector import MyDectection
from threading import Thread

capture_thread = Thread(target = MyDectection().run_dectection(), name='capture_thread')

capture_thread.start()

