// Rename this file to App.jsx for proper JSX support
import React, { useState, useMemo } from "react";
import racesData from "./racesData.json";
import "./App.css";

const fields = [
  { key: "name", label: "Race Name" },
  { key: "location", label: "Location" },
  { key: "registration", label: "Registration Link" },
  { key: "price", label: "Price" },
  { key: "difficulty", label: "Difficulty Level" },
  { key: "summary", label: "Summary/Description" },
  { key: "details", label: "Other Details" }
];

function App() {
  const [filters, setFilters] = useState({});
  const [expanded, setExpanded] = useState({});

  const filteredRaces = useMemo(() => {
    return racesData.filter(race =>
      fields.every(field => {
        const value = filters[field.key];
        if (!value) return true;
        return String(race[field.key] || "")
          .toLowerCase()
          .includes(value.toLowerCase());
      })
    );
  }, [filters]);

  const handleFilterChange = (key, value) => {
    setFilters(f => ({ ...f, [key]: value }));
  };

  const toggleExpand = key => {
    setExpanded(e => ({ ...e, [key]: !e[key] }));
  };

  return (
    <div className="container">
      <h1>Top 1000 Triathlon Races of 2026</h1>
      <div className="filters">
        {fields.map(field => (
          <div key={field.key} className="filter-group">
            <button
              className="expand-btn"
              onClick={() => toggleExpand(field.key)}
              aria-label={`Expand filter for ${field.label}`}
            >
              {expanded[field.key] ? "−" : "+"}
            </button>
            <span className="filter-label">{field.label}</span>
            {expanded[field.key] && (
              <input
                type="text"
                className="filter-input"
                placeholder={`Filter by ${field.label}`}
                value={filters[field.key] || ""}
                onChange={e => handleFilterChange(field.key, e.target.value)}
              />
            )}
          </div>
        ))}
      </div>
      <div className="table-wrapper">
        <table className="races-table">
          <thead>
            <tr>
              {fields.map(field => (
                <th key={field.key}>{field.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filteredRaces.slice(0, 1000).map((race, idx) => (
              <tr key={idx}>
                <td>{race.name}</td>
                <td>{race.location}</td>
                <td>
                  <a href={race.registration} target="_blank" rel="noopener noreferrer">
                    Register
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
