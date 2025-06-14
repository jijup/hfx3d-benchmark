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

| Building ID | Building Name            | Review 1 & Label 0 Removed | Review 2 & Label 0 Removed |
|-------------|--------------------------|-----------------------------|-----------------------------|
| `BLD001`    | Goldberg                 | <input type="checkbox" />  | <input type="checkbox" />  |
| `BLD002`    | MonaCampbell_A           | <input type="checkbox" />  | <input type="checkbox" />  |
| `BLD003`    | MonaCampbell_B           | <input type="checkbox" />  | <input type="checkbox" />  |
| `BLD004`    | Kenneth_Rowe             | <input type="checkbox" />  | <input type="checkbox" />  |
| `BLD005`    | Schulich_School_of_Law   | <input type="checkbox" />  | <input type="checkbox" />  |
| `BLD006`    | VIC_Suites               | <input type="checkbox" />  | <input type="checkbox" />  |
| `BLD007`    | Engineering_Building     | <input type="checkbox" />  | <input type="checkbox" />  |
| `BLD008`    | MonaCampbell_C           | <input type="checkbox" />  | <input type="checkbox" />  |
| `BLD009`    | Halifax_Central_Library  | <input type="checkbox" />  | <input type="checkbox" />  |
| `BLD010`    | Emera_Place              | <input type="checkbox" />  | <input type="checkbox" />  |
| `BLD011`    | Dal_Gerard_Hall          | <input type="checkbox" />  | <input type="checkbox" />  |
| `BLD012`    | The_Margaritta           | <input type="checkbox" />  | <input type="checkbox" />  |
| `BLD013`    | 1801_Hollis_St_1         | <input type="checkbox" />  | <input type="checkbox" />  |
| `BLD014`    | Summit_Place             | <input type="checkbox" />  | <input type="checkbox" />  |
| `BLD015`    | 1801_Hollis_St_2         | <input type="checkbox" />  | <input type="checkbox" />  |
| `BLD016`    | 1801_Hollis_St_3         | <input type="checkbox" />  | <input type="checkbox" />  |
| `BLD017`    | 1801_Hollis_St_4         | <input type="checkbox" />  | <input type="checkbox" />  |
| `BLD018`    | The_Exchange             | <input type="checkbox" />  | <input type="checkbox" />  |
| `BLD019`    | ____________             | <input type="checkbox" />  | <input type="checkbox" />  |
| `BLD020`    | ____________             | <input type="checkbox" />  | <input type="checkbox" />  |

---

### ✅ Example of Filename Breakdown

```
HFX_BLD002_ZEB_CLEAN.las
```

- `HFX`: Halifax.
- `BLD002`: MonaCampbell_A.
- `ZEB`: Collected using ZEB Horizon Scanner.
- `CLEAN`: Cleaned and labeled scan (outliers/noise removed).
