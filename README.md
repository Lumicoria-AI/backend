# Lumicoria AI Backend

This is the backend API for Lumicoria AI, built with FastAPI and PostgreSQL.

## Features

- User authentication with JWT tokens
- User profile management
- User settings management
- Avatar upload and management
- RESTful API endpoints
- PostgreSQL database with SQLAlchemy ORM
- CORS middleware for frontend integration
- Static file serving for user uploads

## Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Virtual environment (recommended)

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the backend directory with the following variables:
```env
POSTGRES_SERVER=localhost
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=lumicoria
SECRET_KEY=your_secret_key
BACKEND_CORS_ORIGINS=["http://localhost:8080","http://localhost:3000"]
```

4. Initialize the database:
```bash
alembic upgrade head
```

5. Run the development server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- Swagger UI documentation: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

### Available Endpoints

#### Authentication
- `POST /api/v1/auth/login` - Login with email and password
- `POST /api/v1/auth/signup` - Create a new user account
- `POST /api/v1/auth/test-token` - Test access token

#### Users
- `GET /api/v1/users/me` - Get current user
- `PUT /api/v1/users/me` - Update current user
- `GET /api/v1/users/me/profile` - Get user profile
- `PUT /api/v1/users/me/profile` - Update user profile
- `GET /api/v1/users/me/settings` - Get user settings
- `PUT /api/v1/users/me/settings` - Update user settings
- `POST /api/v1/users/me/avatar` - Upload user avatar

## Database Schema

### Users Table
- `id` (String, Primary Key)
- `email` (String, Unique)
- `full_name` (String)
- `hashed_password` (String)
- `avatar_url` (String, Nullable)
- `is_active` (Boolean)
- `is_superuser` (Boolean)
- `created_at` (DateTime)
- `updated_at` (DateTime)

### User Profiles Table
- `id` (String, Primary Key)
- `user_id` (String, Foreign Key)
- `job_title` (String, Nullable)
- `company` (String, Nullable)
- `timezone` (String)
- `preferred_language` (String)
- `created_at` (DateTime)
- `updated_at` (DateTime)

### User Settings Table
- `id` (String, Primary Key)
- `user_id` (String, Foreign Key)
- `email_notifications` (Boolean)
- `push_notifications` (Boolean)
- `task_reminders` (Boolean)
- `break_reminders` (Boolean)
- `work_hours_start` (String)
- `work_hours_end` (String)
- `break_interval_minutes` (Integer)
- `break_duration_minutes` (Integer)
- `preferred_ai_model` (String)
- `created_at` (DateTime)
- `updated_at` (DateTime)

## Development

### Running Tests
```bash
pytest
```

### Code Style
The project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting

Run the formatters:
```bash
black .
isort .
flake8
```

### Database Migrations
To create a new migration:
```bash
alembic revision --autogenerate -m "description"
```

To apply migrations:
```bash
alembic upgrade head
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 