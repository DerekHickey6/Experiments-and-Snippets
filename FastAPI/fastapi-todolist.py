from fastapi import FastAPI, HTTPException
from typing import List, Optional
from enum import IntEnum
from pydantic import BaseModel, Field

api = FastAPI()

class Priority(IntEnum):
    LOW = 3
    MEDIUM = 2
    HIGH = 1

# Mandatory base for a todo item
class TodoBase(BaseModel):
    todo_name: str = Field(..., min_length=3, max_length=512, description='Name of the todo')
    todo_description: str = Field(..., description='description of the todo')
    priority: Priority = Field(default=Priority.LOW, description='Priority of the todo')

class TodoCreate(TodoBase):
    pass

class Todo(TodoBase):
    todo_id: int = Field(..., description='Unique identifier of todo')

# Fields are optionally updated - can change just 1 if desired
class TodoUpdate(BaseModel):
    todo_name: Optional[str] = Field(None, min_length=3, max_length=512, description='Name of the todo')
    todo_description: Optional[str] = Field(None, description='description of the todo')
    priority: Optional[Priority] = Field(None, description='Priority of the todo')




# Get - information from server
# Post - add information to server,
# Put - change information on a server,
# Delete - deleted information from server

all_todos = [
    Todo(todo_id=1, todo_name='Sport', todo_description='Go to the gym', priority=Priority.HIGH),
    Todo(todo_id=2, todo_name='Read', todo_description='Read 10 pages', priority=Priority.MEDIUM),
    Todo(todo_id=3, todo_name='Shop', todo_description='Get groceries', priority=Priority.LOW),
    Todo(todo_id=4, todo_name='Study', todo_description='Study for exam', priority=Priority.MEDIUM),
    Todo(todo_id=5, todo_name='Meditate', todo_description='Meditate 20 minutes', priority=Priority.LOW)
]

# Delete a todo
@api.delete('todos/{todo_id}')
def delete_todo(todo_id: int):

    for index, todo in enumerate(all_todos):
        if todo.todo_id == todo_id:
            delete_todo = all_todos.pop(index)
            return delete_todo

    raise HTTPException(status_code=404, detail="todo not found")


# Update name and description of an existing todo
@api.put('/todos/{todo_id}', response_model=Todo)
def update_todo(todo_id: int, updated_todo: TodoUpdate):
    for todo in all_todos:
        if todo.todo_id == todo_id:
            if updated_todo.todo_name is not None:
                todo.todo_name = updated_todo.todo_name
            if updated_todo.todo_description is not None:
                todo.todo_description = updated_todo.todo_description
            if updated_todo.priority is not None:
                todo.priority = updated_todo.priority

            return todo

    raise HTTPException(status_code=404, detail="todo not found")

# Create todos and add them to database
@api.post('/todos', response_model=Todo)
def create_todos(todo: TodoCreate):
    new_todo_id = max(todo.todo_id for todo in all_todos) + 1

    new_todo = Todo(todo_id=new_todo_id,
                    todo_name=todo.todo_name,
                    todo_description=todo.todo_description,
                    priority=todo.priority)

    all_todos.append(new_todo)

    return new_todo


# Path parameter - Locolhost:999/todos/2
# Query parameter - Locolhost:999/todos?first_n=3

# Return specific todos
@api.get('/todos/{todo_id}', response_model=Todo)
def get_todo(todo_id: int):
    for todo in all_todos:
        if todo.todo_id == todo_id:
            return todo

    raise HTTPException(status_code=404, detail="todo not found")

# Retrun all todos
@api.get('/todos', response_model=List[Todo])
def get_todo(first_n: int = None):
    if first_n:
        return all_todos[:first_n]
    else:
        return all_todos


# Asyncronous
@api.get('/calculation')
def calculation():
    # do calculation
    # CPU - doesn't make sense to make Async
    pass
    return ""

@api.get('/getdata')
async def get_data_from_df():
    pass

