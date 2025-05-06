```mermaid
graph TD
    User(User)
    subgraph API_Server [API Server]
        A1[Endpoint analyze] --> B(Start Background Task)
        A2[Endpoint /query] --> C{Invoke Unified Graph}
        A3[Endpoint /analysis/{task_id}] --> D(Read analysis_tasks Dict)
        D --> A3_Resp(Return Task Status)
        C --> A2_Resp(Return Query Response)
        X(Call init_db at startup)
    end

    subgraph Background Task Runner
        B --> E{Invoke Unified Graph (Analysis Flow)}
        E --> F(Get Analysis Result/Summary)
        F --> G(Log Task to DB via database.py)
        G --> H(Update analysis_tasks Dict)
    end

    subgraph Database_Module [Database Module (database.py)]
        I[SQLite DB (analysis_history.db)]
        J(init_db) --> I
        K(log_task_status) --> I
        L(query_tasks) --> I
    end

    subgraph History_Tool_Module [History Tool Module (history_tools.py)]
        M(query_analysis_history_tool: Tool) --> N(_query_analysis_history_func)
        N --> L
    end

    subgraph Unified_Graph [Unified Graph (graph_builder.py)]
        O{Agent Node} -- Uses Tools --> P([Tool List])
        P --> SearchTool(search_tool)
        P --> FileTools(file_tools)
        P --> M[query_analysis_history_tool]
        C --> O
    end

    subgraph Agent_Prompt [Agent Prompt (prompt_template.txt)]
        Q(Instructions for Agent) -- Defines Usage --> M
    end

    User -- "/analyze (industry)" --> A1
    User -- "/query (chat/history question)" --> A2
    User -- "/analysis/{task_id} (check status)" --> A3

    G --> K

    style AgentNode fill:#f9f,stroke:#333,stroke-width:2px
    style M fill:#ccf,stroke:#333,stroke-width:2px
    style I fill:#ff9,stroke:#333,stroke-width:2px
