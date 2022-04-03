let interval = setInterval( () => {
    //check to see if error class is present on screen
    if (document.getElementById('cell-1bbNPWmojQke').classList.contains('code-has-output')) { //Checking to see if HTMLCollection array returned has any elements in it
        console.log('Error found')
        //define disconnect function
        const disconnectColab = () => {
            function simulate(element, eventName)
            {
                var options = extend(defaultOptions, arguments[2] || {});
                var oEvent, eventType = null;
            
                for (var name in eventMatchers)
                {
                    if (eventMatchers[name].test(eventName)) { eventType = name; break; }
                }
            
                if (!eventType)
                    throw new SyntaxError('Only HTMLEvents and MouseEvents interfaces are supported');
            
                if (document.createEvent)
                {
                    oEvent = document.createEvent(eventType);
                    if (eventType == 'HTMLEvents')
                    {
                        oEvent.initEvent(eventName, options.bubbles, options.cancelable);
                    }
                    else
                    {
                        oEvent.initMouseEvent(eventName, options.bubbles, options.cancelable, document.defaultView,
                        options.button, options.pointerX, options.pointerY, options.pointerX, options.pointerY,
                        options.ctrlKey, options.altKey, options.shiftKey, options.metaKey, options.button, element);
                    }
                    element.dispatchEvent(oEvent);
                }
                else
                {
                    options.clientX = options.pointerX;
                    options.clientY = options.pointerY;
                    var evt = document.createEventObject();
                    oEvent = extend(evt, options);
                    element.fireEvent('on' + eventName, oEvent);
                }
                return element;
            }
            
            function extend(destination, source) {
                for (var property in source)
                  destination[property] = source[property];
                return destination;
            }
            
            var eventMatchers = {
                'HTMLEvents': /^(?:load|unload|abort|error|select|change|submit|reset|focus|blur|resize|scroll)$/,
                'MouseEvents': /^(?:click|dblclick|mouse(?:down|up|over|move|out))$/
            }
            var defaultOptions = {
                pointerX: 0,
                pointerY: 0,
                button: 0,
                ctrlKey: false,
                altKey: false,
                shiftKey: false,
                metaKey: false,
                bubbles: true,
                cancelable: true
            }
            
            document.getElementById('top-toolbar').querySelectorAll('colab-connect-button')[0].shadowRoot.getElementById('connect-dropdown').click();
            setTimeout(() => {
                let el_ = document.getElementsByClassName('goog-menu-vertical')[0]['children'][5].getElementsByClassName('goog-menuitem-content')[0].parentElement
                simulate(el_, "mousedown");
                simulate(el_, "mouseup");
            }, 1000);
            
            setTimeout(() => {
                //const terminateButton = document.querySelector("body > colab-dialog > paper-dialog > colab-sessions-dialog").shadowRoot.querySelector("div.dialog-main-content > div.sessions-content.layout.vertical > div.dialog-table > colab-session > div.button-action-column > paper-icon-button");
                const terminateButton = document.querySelector("body > colab-dialog > paper-dialog > colab-sessions-dialog").shadowRoot.querySelector("div.dialog-main-content > div.sessions-content.layout.vertical > div.dialog-table > colab-session > div.button-action-column > paper-icon-button")
                if (terminateButton) {
                    terminateButton.click();
                }
            }, 3000);
            
            setTimeout(() => {
                const confirmTerminationButton = document.querySelector("#ok");
                if (confirmTerminationButton){
                        confirmTerminationButton.click()
                }
            }, 5000);   
        }
        //call disconnect function
        disconnectColab();
        clearInterval(interval);
    } else {
        console.log('Still training')
    }
},8000)