### 📄 Naming Convention and Building Index

#### 🔹 HFX3D File Naming Format

Each file is named using the format:

```
HFX_<BuildingID>_<SensorType>_<Status>.las
```

#### 🔹 Components

| Part             | Description                                           |
|------------------|-------------------------------------------------------|
| `HFX`            | City code for Halifax, Nova Scotia.                   |
| `<BuildingID>`   | Unique identifier for each building (e.g., `BLD001`). |
| `<SensorType>`   | LiDAR device used.                                    |
| `<Status>`       | The status of the data - Clean.                       |
| `.las`           | File extension.                                       |

---

### 🏡 Building Index

| Building ID | Building Name         |
|-------------|-----------------------|
| `BLD001`    | Goldberg              |
| `BLD002`    | MonaCampbell_A        |
| `BLD003`    | MonaCampbell_B        |
| `BLD004`    | Kenneth_Rowe          |
| `BLD005`    | Schulich_School_of_Law|
| `BLD006`    | VIC_Suites            |
| `BLD007`    | *(Not assigned yet)*  |
| `BLD008`    | *(Not assigned yet)*  |


---

### ✅ Example Filename Breakdown

```
HFX_BLD002_ZEB_CLEAN.las
```

- `HFX`: Halifax.
- `BLD002`: MonaCampbell_A.
- `ZEB`: Collected using ZEB Horizon Scanner.
- `CLEAN`: Cleaned and labeled scan (outliers/noise removed).
