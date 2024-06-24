window.addEventListener("load", function () {
    const getFamily = () =>
        document.querySelector("[type=radio][name=model-family]:checked").value;
    const getTask = () =>
        document.querySelector("[type=radio][name=task]:checked").value;

    const testCasesEl = document.getElementById("testCases");
    const promptsEl = document.getElementById("prompts");
    const modelPromptEl = document.getElementById("modelPrompt");
    const modelTestCaseEl = document.getElementById("modelTestCase");
    const modelInfoEl = document.getElementById("modelInfo");
    const modelInputEl = document.getElementById("modelInput");
    const modelInputDisplayEl = document.getElementById("modelInputDisplay");
    const modelOutputEl = document.getElementById("modelOutput");

    modelInputDisplay.value = "";
    modelOutputEl.value = "";
    
    const appTab1 = document.getElementById("appTab1");
    const appTab2 = document.getElementById("appTab2");
    const appTab3 = document.getElementById("appTab3");

    appTab1.checked = true;

    const clearChildren = el => {
        while(el.firstChild) {
            el.removeChild(el.firstChild);
        }
    };

    const popup = (header, text) => {
        document.getElementById("modalPopup").checked = true;
        document.getElementById("popupHeader").textContent = header;
        document.getElementById("popupContent").textContent = text;
    };
    const cancelPopup = () => {
        document.getElementById("modalPopup").checked = false;
    };

    const updateModelInput = () => {
        modelInputEl.value = modelPromptEl.value.replace("{prompt}", modelTestCase.value);
    };
    modelPromptEl.addEventListener("change", updateModelInput);
    modelTestCaseEl.addEventListener("change", updateModelInput);

    const createStackElement = params => {
        let { name, value, label, change, checked } = params;
        let stackElement = document.createElement("label");
        stackElement.className = "stack";
        let input = document.createElement("input");
        input.type = "radio";
        input.name = name;
        input.value = value;
        input.checked = checked;
        input.addEventListener("change", change);
        let span = document.createElement("span");
        span.className = "button toggle";
        span.textContent = label;
        stackElement.appendChild(input);
        stackElement.appendChild(span);
        return stackElement;
    };

    // if refresh during popup, it will carry on
    cancelPopup();
    
    document.getElementById("loadCases").addEventListener("click", async () => {
        popup("Loading...", "Waiting for response from server");
        const requestData = {
            family: getFamily(),
            task: getTask(),
        };
        const response = await fetch("/load_cases", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(requestData),
        });

        if(!response.ok) {
            popup("Error!", `Bad network response: ${response.statusText}`);
            return;
        }

        const responseData = await response.json();

        let { prompts, test_cases: testCases } = responseData;
        
        // make stack for prompts
        clearChildren(promptsEl);
        const handlePromptChange = function () {
            modelPromptEl.value = prompts[this.value];
            updateModelInput();
        };
        prompts.forEach((prompt, idx) => {
            let stackElement = createStackElement({
                name: "prompt",
                value: idx,
                label: `prompt${idx}`,
                change: handlePromptChange,
                checked: idx === 0,
            });
            promptsEl.appendChild(stackElement);
        });

        // make stack for test cases
        clearChildren(testCasesEl);
        const handleTestCaseChange = function () {
            modelTestCase.value = testCases[this.value];
            updateModelInput();
        };
        testCases.forEach((testCase, idx) => {
            let stackElement = createStackElement({
                name: "testCase",
                value: idx,
                label: (idx + 1).toString().padStart(3, 0) + ": " + testCases[idx],
                change: handleTestCaseChange,
                checked: idx === 0,
            });
            testCasesEl.appendChild(stackElement);
        });

        for(let el of document.querySelectorAll(".if-model-loaded")) {
            el.classList.remove("hidden");
        }
        
        modelInfoEl.textContent = `CodeGen1-multi-${requestData.family}, ${requestData.task}`;
        handlePromptChange.call({ value: 0 });
        handleTestCaseChange.call({ value: 0 });
        cancelPopup();
        appTab2.checked = true;
    });

    document.getElementById("runModel").addEventListener("click", async () => {
        modelInputDisplayEl.value = modelInputEl.value;
        modelOutputEl.value = "";
        appTab3.checked = true;
        const requestData = {
            family: getFamily(),
            task: getTask(),
            input: modelInputEl.value,
        };
        const response = await fetch("/run_single", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(requestData),
        });

        if(!response.ok) {
            popup("Error!", `Bad network response: ${response.statusText}`);
            return;
        }

        const responseData = await response.json();
        
        console.log(responseData);
        if(responseData.output === "") {
            modelOutputEl.value = "!! MODEL DID NOT PRODUCE ANY TOKENS IN RESPONSE TO PROMPTING (check server console) !!";
        }
        else {
            modelOutputEl.value = responseData.output;
        }
    });
});
