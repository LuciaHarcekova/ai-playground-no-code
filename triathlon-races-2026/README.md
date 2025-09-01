# Copilot Prompt - Single Page Application

🚀 From zero to a fully functional web page in a few minutes, let GitHub Copilot do the heavy lifting while you focus on customization!  

No prior coding experience required! Build a simple web application to showcase **Triathlon Races for 2026** with details such as location, prices, difficulty, country, official website, and more. Built with **React/Next.js**, styled with **Tailwind CSS**, and powered by **GitHub Copilot** for rapid development.

---

## Steps to Run the App

1. Clone the repository:
  ```bash
  git clone <repo-url>
  cd triathlon-races-2026/web
  ```
2. Install dependencies:
  ```bash
  npm install
  ```
3. Start the development server:
  ```bash
  npm run dev
  ```
4. Open the app in your browser at the URL shown in the terminal (usually http://localhost:5173).

---

## Steps to generate Single Page Application - 🏊‍♂️🚴‍♀️🏃 Triathlon Races 2026 Webpage


1. 🔧 Setup Environment  
2. 🤖 Ask Copilot to generate page  
3. 🛠 Fixing  
4. ✨ Adjust based on your recommendation  

---

## 🔧 1. Setup Environment

### 🤖 Set up Copilot

**A. Install VS Code Extensions**  
- Install **GitHub Copilot** and **GitHub Copilot Chat** from the Extensions tab.  
- Copilot Chat gives you an interactive panel for generating larger code snippets.  

**B. Sign in to GitHub**  
- After installation, sign in to authorize VS Code.  
- Ensure your GitHub account has **Copilot access**.  

**C. Verify Installation**  
- You should see the **Copilot icon** in the bottom status bar.  
- Test by typing a comment in any code file, e.g.:
  // Create a React component that displays a table of races


Copilot should start suggesting completions automatically.

**Resources**

* [GitHub Copilot Quickstart](https://docs.github.com/en/copilot/get-started/quickstart)
* [VS Code Copilot Setup](https://code.visualstudio.com/docs/copilot/setup-simplified)

### Create a project folder

```bash
mkdir triathlon-races-2026
cd triathlon-races-2026
```

---

## 🤖 2. Ask Copilot to Generate Page

Use this prompt in **Copilot Chat**:

```
Create a new project named "triathlon-races-2026".
Inside this project, add all necessary resources to build a single web page that displays a table in the center of the page containing the top 1000 triathlon races of different distances for the year 2026.
The race data should be gathered from search results of triathlon races.
The table must include the following fields: Race name, Location, Date, Registration link, Price, Difficulty level, Summary/description and any other important details.
At the top of the table, add a filter for each field.
Filters should be elegant and expandable, not bulky or overwhelming.
Style the page using a blue, green, and gray color palette.
Also, add a .gitignore file to the project to exclude resources that are not required for other users to run the solution after cloning from GitHub.
After implementing the page, run it locally.
```

---

## 🛠 3. Fixing

I tried different prompts to create a single web page. Some attempts ended up with errors when trying to launch the app. Thanks to Copilot, I was able to resolve most code issues.

The main struggle I faced was **gathering outside data**.
Copilot generated only **6 races** instead of 1000.

* Fix Prompt: *"You have listed only 6 races, why not 1000? Fill in another 1000 legitimate races for 2026."*
* Response: *"Sorry, I can't assist with that."*

✅ **Solution:** I used ChatGPT to generate an Excel/JSON file with 1000 races for 2026. This worked perfectly — I quickly gathered the data and updated it in `racesData.json`.

---

## ✨ 4. Adjust Based on Recommendations

Once Copilot generates the page, refine it by asking:

* For the **Country filter**, allow both free text input and, below it, show a list of existing country options from the table.
* Do the same for the **Difficulty Level filter**.
* For **Price**, add a range selector (minimum = lowest price in the table, maximum = highest).
* At the top of the table, add **sorting** for `Price`, `Location`, `Name`, and `Difficulty`.
* Remove the filter on the **Registration Link** field.
* Rename column **"Registration Link"** to **"Link"**.
* Replace column **"Other Details"** with **"Distance"**.
* For **Date**, use a date selector and display it as a calendar.
* For filterable fields, display available options as a **dropdown below the input field**. Hide them when the user is not typing anymore (instead of showing them inline at all times).


