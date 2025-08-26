# CloudCasting Backend API

[![ease of contribution: easy](https://img.shields.io/badge/ease%20of%20contribution:%20easy-32bd50)](https://github.com/openclimatefix#how-easy-is-it-to-get-involved) 
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

A FastAPI-based backend service for processing and serving satellite cloud forecasting data. This API provides endpoints for downloading cloudcasting data from S3, converting Zarr files to GeoTIFF format, and serving the processed data for visualization in mapping applications.

## Features

- **Satellite Data Processing**: Downloads and processes cloudcasting forecast data from S3
- **Format Conversion**: Converts Zarr format to GeoTIFF for mapping applications
- **Background Processing**: Asynchronous data download and conversion
- **RESTful API**: Clean API endpoints for data access and status monitoring
- **Authentication**: JWT-based authentication with Auth0 integration for protected endpoints
- **Static File Serving**: Direct access to processed GeoTIFF layers
- **Data Information**: Detailed metadata about processed data and timing information

## Running the service

### Configuration

The application is configured via environment variables. For the complete list of environment variables, see the [`.env.example`](.env.example) file.

**Required Environment Variables:**
```bash
# S3 Configuration (Required)
CLOUDCASTING_BACKEND_S3_BUCKET_NAME=your-bucket-name
CLOUDCASTING_BACKEND_S3_REGION_NAME=us-east-1
CLOUDCASTING_BACKEND_S3_ACCESS_KEY_ID=your-access-key-id
CLOUDCASTING_BACKEND_S3_SECRET_ACCESS_KEY=your-secret-access-key

# Auth0 Configuration
AUTH0_DOMAIN=your-tenant.auth0.com
AUTH0_API_AUDIENCE=your-api-audience
```

Copy `.env.example` to `.env` and configure the values for your environment:
```bash
$ cp .env.example .env
# Edit .env with your configuration
```

### Using Docker

Build and run locally using Docker:


For development with automatic reload:

```sh
$ docker-compose up --build
```

### Using python(v3.11.x)

Clone the repository and create a new virtual environment with your favorite environment manager.
Install the dependencies with Poetry:

```bash
$ poetry install
```

The service is then runnable via the command:
```bash
$ poetry run python -m cloudcasting_backend
```

You should see the following output:

```shell
INFO:     Started server process [87312]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

The API should then be accessible at `http://localhost:8000`,
and the docs at `http://localhost:8000/api/docs`.

## API Endpoints

### Cloudcasting Data

- `GET /api/cloudcasting/status` - Get status of cloudcasting data availability
- `POST /api/cloudcasting/trigger-download` - Trigger background download of latest data  
- `GET /api/cloudcasting/download-status` - Get current status of background download processes
- `GET /api/cloudcasting/data-info` - Get detailed information about processed data including timing and metadata
- `GET /api/cloudcasting/layers` - Get list of available channels and forecast steps
- `GET /api/cloudcasting/layers/{channel}/{step}.tif` - Download specific GeoTIFF layer


### Data Information Endpoint

The `/api/cloudcasting/data-info` endpoint provides comprehensive information about the processed data:

```json
{
  "file_exists": true,
  "init_time": null,
  "forecast_steps": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11
  ],
  "variables": [
    "VIS006",
    "WV_062",
    "WV_073",
    "VIS008",
    "IR_039",
    "IR_108",
    "IR_120",
    "IR_134",
    "IR_087",
    "IR_016",
    "IR_097"
  ],
  "file_size_mb": 70.29,
  "last_modified": "2025-08-26T09:43:26+00:00",
  "time_range": {
    "last_processed": "2025-08-26T09:43:26+00:00",
    "data_source": "S3 download timestamp",
    "total_forecast_steps": 12,
    "available_variables": 11
  },
  "error": null
}}
```

This endpoint:
- Reports on processed GeoTIFF layers (not temporary zarr files)
- Uses timestamps from the data processing pipeline
- Provides metadata for frontend applications
- Enables monitoring of data freshness and availability

## Development

Clone the repository and create a new environment with your favorite environment manager.
Install all the dependencies including development tools:

```bash
$ poetry install
```

You can run the service with auto-reload for development:
```bash
$ CLOUDCASTING_BACKEND_RELOAD=true poetry run python -m cloudcasting_backend
```

### Code Quality

This project uses Black for code formatting and pytest for testing.

Install pre-commit hooks:
```bash
$ poetry run pre-commit install
```

Run code formatting:
```bash
$ poetry run black cloudcasting_backend tests
```

## Running Tests

Make sure you have the development dependencies installed:

```bash
$ poetry install
```

Then run the tests using pytest:
```bash
$ poetry run pytest
```

For test coverage:
```bash
$ poetry run pytest --cov=cloudcasting_backend --cov-report=html
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CLOUDCASTING_BACKEND_HOST` | `0.0.0.0` | Server host interface |
| `CLOUDCASTING_BACKEND_PORT` | `8000` | Server port |
| `CLOUDCASTING_BACKEND_WORKERS_COUNT` | `1` | Number of worker processes |
| `CLOUDCASTING_BACKEND_RELOAD` | `false` | Enable auto-reload for development |
| `CLOUDCASTING_BACKEND_LOG_LEVEL` | `INFO` | Logging level |
| `CLOUDCASTING_BACKEND_ENVIRONMENT` | `dev` | Environment identifier |
| `CLOUDCASTING_BACKEND_S3_BUCKET_NAME` | - | **Required**: S3 bucket name |
| `CLOUDCASTING_BACKEND_S3_REGION_NAME` | - | **Required**: AWS region |
| `CLOUDCASTING_BACKEND_S3_ACCESS_KEY_ID` | - | **Required**: AWS access key |
| `CLOUDCASTING_BACKEND_S3_SECRET_ACCESS_KEY` | - | **Required**: AWS secret key |
| `CLOUDCASTING_BACKEND_SENTRY_DSN` | - | Optional: Sentry error tracking DSN |
| `AUTH0_DOMAIN` | - | Auth0 domain for JWT authentication |
| `AUTH0_API_AUDIENCE` | - | Auth0 API audience for JWT validation |

## Known Issues

- Large dataset processing may require significant memory and processing time
- S3 access requires proper AWS credentials and permissions
- Some projections may have edge cases in coordinate transformation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://suvanbanerjee.github.io"><img src="https://avatars.githubusercontent.com/u/104707806?v=4?s=100" width="100px;" alt="Suvan Banerjee"/><br /><sub><b>Suvan Banerjee</b></sub></a><br /><a href="https://github.com/openclimatefix/cloudcasting-backend/commits?author=suvanbanerjee" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

---

*Part of the [Open Climate Fix](https://github.com/orgs/openclimatefix/people) community.*

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)