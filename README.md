"Why is there no export button?" - Charles Beckwith

This is a simple Python script that you can use to export Ollama Desktop chats from the local SQLite database. 

Place the Python file in the same folder as your Ollama SQL files. 

On Windows, that's usually C:\Users\username\AppData\Local\Ollama

There are detailed instructions at the top of the file, but basically, put it in the same folder with the SQL file and run "python .\ollama_export.py"

You will need to have Python installed first.

It will create a chat_export folder, with sub-folders for each date when a chat was started, and place the transcript and any attachemnts you uploaded into Ollama into those date folders. 

You should close Ollama Desktop before you run this, and give it a few secconds to save the current chat to the right SQL file (db.sqlite). The program checks to see if the db.sqlite-shm file is 0KB before it will write anything. Closing the application dumps db.sqlite-shm into db.sqlite, allowing the program to run.
