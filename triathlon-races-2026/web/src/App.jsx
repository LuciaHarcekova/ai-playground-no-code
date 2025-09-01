import React, { useState, useMemo, useRef } from "react";
import racesData from "./racesData.json";
import "./App.css";

const fields = [
  { key: "name", label: "Race Name", sortable: true },
  { key: "location", label: "Location", sortable: true },
  { key: "date", label: "Date" },
  { key: "registration", label: "Link" },
  { key: "price", label: "Price", sortable: true },
  { key: "difficulty", label: "Difficulty Level", sortable: true },
  { key: "summary", label: "Summary/Description" },
  { key: "details", label: "Distance" }
];

function getCountryOptions(data) {
  const countries = new Set();
  data.forEach(race => {
    const loc = race.location || "";
    const country = loc.split(",").pop().trim();
    if (country) countries.add(country);
  });
  return Array.from(countries).sort();
}

function getDifficultyOptions(data) {
  return Array.from(new Set(data.map(r => r.difficulty))).sort();
}

function getPriceRange(data) {
  const prices = data.map(r => parseFloat((r.price || "").replace(/[^\d.]/g, ""))).filter(Number.isFinite);
  return [Math.min(...prices), Math.max(...prices)];
}

function App() {
  const [filters, setFilters] = useState({});
  const [expanded, setExpanded] = useState({});
  const [sort, setSort] = useState({ key: null, asc: true });
  const [countryDropdownOpen, setCountryDropdownOpen] = useState(false);
  const [difficultyDropdownOpen, setDifficultyDropdownOpen] = useState(false);
  const countryInputRef = useRef(null);
  const difficultyInputRef = useRef(null);

  // For date filter, get all unique dates
  const dateOptions = useMemo(() => Array.from(new Set(racesData.map(r => r.date))).sort(), []);

  const countryOptions = useMemo(() => getCountryOptions(racesData), []);
  const difficultyOptions = useMemo(() => getDifficultyOptions(racesData), []);
  const [minPrice, maxPrice] = useMemo(() => getPriceRange(racesData), []);

  const filteredRaces = useMemo(() => {
    return racesData.filter(race => {
      // Country filter
      if (filters.location) {
        const loc = race.location || "";
        const country = loc.split(",").pop().trim();
        if (!country.toLowerCase().includes(filters.location.toLowerCase()) &&
            !loc.toLowerCase().includes(filters.location.toLowerCase())) return false;
      }
      // Difficulty filter
      if (filters.difficulty) {
        if (!(race.difficulty || "").toLowerCase().includes(filters.difficulty.toLowerCase())) return false;
      }
      // Price range filter
      if (filters.priceMin || filters.priceMax) {
        const price = parseFloat((race.price || "").replace(/[^\d.]/g, ""));
        if (filters.priceMin && price < filters.priceMin) return false;
        if (filters.priceMax && price > filters.priceMax) return false;
      }
      // Date filter
      if (filters.date) {
        if (race.date !== filters.date) return false;
      }
      // Other filters
      for (const field of fields) {
        if (["location", "difficulty", "price", "registration", "date"].includes(field.key)) continue;
        const value = filters[field.key];
        if (!value) continue;
        if (!String(race[field.key] || "").toLowerCase().includes(value.toLowerCase())) return false;
      }
      return true;
    });
  }, [filters]);

  const sortedRaces = useMemo(() => {
    if (!sort.key) return filteredRaces;
    return [...filteredRaces].sort((a, b) => {
      let va = a[sort.key], vb = b[sort.key];
      if (sort.key === "price") {
        va = parseFloat((va || "").replace(/[^\d.]/g, ""));
        vb = parseFloat((vb || "").replace(/[^\d.]/g, ""));
      }
      if (va === undefined || vb === undefined) return 0;
      if (va < vb) return sort.asc ? -1 : 1;
      if (va > vb) return sort.asc ? 1 : -1;
      return 0;
    });
  }, [filteredRaces, sort]);

  const handleFilterChange = (key, value) => {
    setFilters(f => ({ ...f, [key]: value }));
  };

  const toggleExpand = key => {
    setExpanded(e => ({ ...e, [key]: !e[key] }));
  };

  const handleSort = key => {
    setSort(s => s.key === key ? { key, asc: !s.asc } : { key, asc: true });
  };

  return (
    <div className="container">
      <h1>Top 1000 Triathlon Races of 2026</h1>
      <div className="filters">
        {/* Country filter */}
        <div className="filter-group">
          <button className="expand-btn" onClick={() => toggleExpand("location")} aria-label="Expand filter for Country">{expanded["location"] ? "−" : "+"}</button>
          <span className="filter-label">Country</span>
          <input
            type="text"
            value={filters.location || ""}
            ref={countryInputRef}
            onFocus={() => setCountryDropdownOpen(true)}
            onBlur={() => setTimeout(() => setCountryDropdownOpen(false), 150)}
            onChange={e => {
              handleFilterChange("location", e.target.value);
              setCountryDropdownOpen(true);
            }}
            placeholder="Type or select country"
            className="filter-input"
          />
          {countryDropdownOpen && (
            <div className="filter-dropdown">
              {(filters.location ? countryOptions.filter(c => c.toLowerCase().includes(filters.location.toLowerCase())) : countryOptions)
                .map(c => (
                  <div
                    key={c}
                    className="filter-option"
                    onMouseDown={() => handleFilterChange("location", c)}
                  >
                    {c}
                  </div>
                ))}
            </div>
          )}
        </div>
        {/* Difficulty filter */}
        <div className="filter-group">
          <button className="expand-btn" onClick={() => toggleExpand("difficulty")} aria-label="Expand filter for Difficulty">{expanded["difficulty"] ? "−" : "+"}</button>
          <span className="filter-label">Difficulty Level</span>
          <input
            type="text"
            value={filters.difficulty || ""}
            ref={difficultyInputRef}
            onFocus={() => setDifficultyDropdownOpen(true)}
            onBlur={() => setTimeout(() => setDifficultyDropdownOpen(false), 150)}
            onChange={e => {
              handleFilterChange("difficulty", e.target.value);
              setDifficultyDropdownOpen(true);
            }}
            placeholder="Type or select difficulty"
            className="filter-input"
          />
          {difficultyDropdownOpen && (
            <div className="filter-dropdown">
              {(filters.difficulty ? difficultyOptions.filter(d => d.toLowerCase().includes(filters.difficulty.toLowerCase())) : difficultyOptions)
                .map(d => (
                  <div
                    key={d}
                    className="filter-option"
                    onMouseDown={() => handleFilterChange("difficulty", d)}
                  >
                    {d}
                  </div>
                ))}
            </div>
          )}
        </div>
        {/* Price range filter */}
        <div className="filter-group">
          <button className="expand-btn" onClick={() => toggleExpand("price")} aria-label="Expand filter for Price">{expanded["price"] ? "−" : "+"}</button>
          <span className="filter-label">Price Range</span>
          {expanded["price"] && (
            <div style={{ display: "flex", gap: 8 }}>
              <input type="number" className="filter-input" min={minPrice} max={maxPrice} placeholder={`Min (${minPrice})`} value={filters.priceMin || ""} onChange={e => handleFilterChange("priceMin", e.target.value)} />
              <span>to</span>
              <input type="number" className="filter-input" min={minPrice} max={maxPrice} placeholder={`Max (${maxPrice})`} value={filters.priceMax || ""} onChange={e => handleFilterChange("priceMax", e.target.value)} />
            </div>
          )}
        </div>
        {/* Date filter as calendar only */}
        <div className="filter-group">
          <button className="expand-btn" onClick={() => toggleExpand("date")} aria-label="Expand filter for Date">{expanded["date"] ? "−" : "+"}</button>
          <span className="filter-label">Date</span>
          {expanded["date"] && (
            <input
              type="date"
              className="filter-input"
              value={filters.date || ""}
              onChange={e => handleFilterChange("date", e.target.value)}
              style={{ minWidth: 140 }}
            />
          )}
        </div>
        {/* Other filters except registration/link, location, difficulty, price, date */}
        {fields.filter(f => !["location", "difficulty", "price", "registration", "date"].includes(f.key)).map(field => {
          const [dropdownOpen, setDropdownOpen] = useState(false);
          const inputRef = useRef(null);
          const options = Array.from(new Set(racesData.map(r => r[field.key]).filter(Boolean))).sort();
          return (
            <div key={field.key} className="filter-group" style={{ position: "relative" }}>
              <button className="expand-btn" onClick={() => toggleExpand(field.key)} aria-label={`Expand filter for ${field.label}`}>{expanded[field.key] ? "−" : "+"}</button>
              <span className="filter-label">{field.label}</span>
              {expanded[field.key] && (
                <>
                  <input
                    type="text"
                    className="filter-input"
                    placeholder={`Filter by ${field.label}`}
                    value={filters[field.key] || ""}
                    ref={inputRef}
                    onFocus={() => setDropdownOpen(true)}
                    onBlur={() => setTimeout(() => setDropdownOpen(false), 150)}
                    onChange={e => {
                      handleFilterChange(field.key, e.target.value);
                      setDropdownOpen(true);
                    }}
                  />
                  {dropdownOpen && (
                    <div className="filter-dropdown">
                      {(filters[field.key] ? options.filter(opt => String(opt).toLowerCase().includes(filters[field.key].toLowerCase())) : options)
                        .map(opt => (
                          <div
                            key={opt}
                            className="filter-option"
                            onMouseDown={() => {
                              handleFilterChange(field.key, opt);
                              setDropdownOpen(false);
                              inputRef.current.blur();
                            }}
                          >
                            {opt}
                          </div>
                        ))}
                    </div>
                  )}
                </>
              )}
            </div>
          );
        })}
      </div>
      <div className="table-wrapper">
        <table className="races-table">
          <thead>
            <tr>
              {fields.map(field => (
                <th key={field.key}>
                  {field.label}
                  {field.sortable && (
                    <button className="sort-btn" onClick={() => handleSort(field.key)} style={{ background: "none", border: "none", padding: 0, marginLeft: 4, cursor: "pointer" }}>
                      {sort.key === field.key ? (
                        sort.asc ? (
                          <svg width="16" height="16" viewBox="0 0 16 16" style={{ verticalAlign: "middle" }}><path d="M8 6l-3 3h6z" fill="#1976d2"/><path d="M8 10l3-3H5z" fill="#1976d2"/></svg>
                        ) : (
                          <svg width="16" height="16" viewBox="0 0 16 16" style={{ verticalAlign: "middle" }}><path d="M8 10l-3-3h6z" fill="#1976d2"/><path d="M8 6l3 3H5z" fill="#1976d2"/></svg>
                        )
                      ) : (
                        <svg width="16" height="16" viewBox="0 0 16 16" style={{ verticalAlign: "middle" }}><path d="M4 8h8" stroke="#90caf9" strokeWidth="2" strokeLinecap="round"/><path d="M6 6l-2 2 2 2" stroke="#90caf9" strokeWidth="2" strokeLinecap="round" fill="none"/><path d="M10 6l2 2-2 2" stroke="#90caf9" strokeWidth="2" strokeLinecap="round" fill="none"/></svg>
                      )}
                    </button>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sortedRaces.slice(0, 1000).map((race, idx) => (
              <tr key={idx}>
                <td>{race.name}</td>
                <td>{race.location}</td>
                <td>{race.date}</td>
                <td>
                  <a href={race.registration} target="_blank" rel="noopener noreferrer">
                    Link
                  </a>
                </td>
                <td>{race.price}</td>
                <td>{race.difficulty}</td>
                <td>{race.summary}</td>
                <td>{race.details}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default App;
