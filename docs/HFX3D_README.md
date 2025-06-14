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

| Building ID | Building Name            |
|-------------|--------------------------|
| `BLD001`    | Goldberg                 |
| `BLD002`    | MonaCampbell_A           |
| `BLD003`    | MonaCampbell_B           |
| `BLD004`    | Kenneth_Rowe             |
| `BLD005`    | Schulich_School_of_Law   |
| `BLD006`    | VIC_Suites               |
| `BLD007`    | Engineering_Building     |
| `BLD008`    | MonaCampbell_C           |
| `BLD009`    | Halifax_Central_Library  |
| `BLD010`    | Emera_Place              |
| `BLD011`    | Dal_Gerard_Hall          |
| `BLD012`    | The_Margaritta           |
| `BLD013`    | 1801_Hollis_St_1         |
| `BLD014`    | Summit_Place             |
| `BLD015`    | 1801_Hollis_St_2         |
| `BLD016`    | 1801_Hollis_St_3         |
| `BLD017`    | 1801_Hollis_St_4         |
| `BLD018`    | The_Exchange             |
| `BLD019`    | ____________             |
| `BLD020`    | ____________             |

---

### ✅ Example of Filename Breakdown

```
HFX_BLD002_ZEB_CLEAN.las
```

- `HFX`: Halifax.
- `BLD002`: MonaCampbell_A.
- `ZEB`: Collected using ZEB Horizon Scanner.
- `CLEAN`: Cleaned, labeled, and reviewed scan (outliers and noise removed).
