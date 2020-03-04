let hitId = 0;

/* Parameters */
// let rootPath = "https://roi-disruption.s3.amazonaws.com/scene_behavioural/";
let rootPath = "";
let catchFreq = 5;

/* Globals */
var trials = [];
var curTrial = 0;
var curResponse = null;
var nTraining;
var trialStartTime;
var experimentStartTime;
var training = true;
var canProceed = true;

/* Responses */
var responses = [];
var scene1s = [];
var scene2s = [];
var img1s = [];
var img2s = [];
var isCatchs = [];
var reactionTimes = [];

function trialDone() {
    if (!training) {
        // Record the response
        responses.push(curResponse);

        // Record what stimuli that were displayed
        trial = trials[curTrial];
        scene1s.push(trial["scene1"]);
        scene2s.push(trial["scene2"]);
        img1s.push(trial["img1"]);
        img2s.push(trial["img2"]);

        // Note whether or not this was a catch trial
        if (trial["type"] === "catch") {
            isCatchs.push(1);
        } else {
            isCatchs.push(0);
        }

        // Record the reaction time
        var trialEndTime = new Date();
        var rt = trialEndTime - trialStartTime;
        reactionTimes.push(rt);
    }

    if (curTrial === nTraining - 1) {
        training = false;
    }

    curTrial++;

    // Finished experiment
    if (curTrial >= trials.length) {
        doneExperiment();
        return;
    }

    curResponse = null;
    trialBegin();
}

function trialBegin() {
    trialStartTime = new Date();
    $("#trialImg1").prop("src", trials[curTrial]["img1Data"].src);
    $("#trialImg2").prop("src", trials[curTrial]["img2Data"].src);
}

function finishedTraining() {
    canProceed = false;
    $("#trainEndWarning").show();
    $("#proceedExperiment").click(function () {
        canProceed = true;
        $("#trainEndWarning").hide();
        $('#nextTrialMessage').show();
    });
}

function doneExperiment() {
    exportData();
    $("#trial").hide();
    $(document).unbind("keydown.responded");
    $(document).unbind("keydown.nextTrial");
    $("#submitButton").show();
}

function giveFeedback() {
    $('#feedback').show();
    if (trials[curTrial]["scene1"] === trials[curTrial]["scene2"]) {
        if (curResponse == 'same') {
            $('#feedbackAnswer').html("Correct! These are from the SAME room");
            $('#feedbackAnswer').css("color", "green");
        } else {
            $('#feedbackAnswer').html("Incorrect. These are from the SAME room");
            $('#feedbackAnswer').css("color", "red");
        }
    } else {
        if (curResponse == 'different') {
            $('#feedbackAnswer').html("Correct! These are from DIFFERENT rooms");
            $('#feedbackAnswer').css("color", "green");
        } else {
            $('#feedbackAnswer').html("Incorrect. These are from DIFFERENT rooms");
            $('#feedbackAnswer').css("color", "red");
        }
    }
}

function startExperiment() {
    experimentStartTime = new Date();
    $("#startExperiment").hide();
    $("#instructionsContainer").hide();
    $("#trial").show();

    // Click events

    // User has selected a response (pressed a key)
    $(document).bind("keydown.responded", function (event) {
        // Check if the key corresponds to a valid response
        if (event.which != 70 && event.which != 74) {
            return;
        }

        // If this is the last training image, give a warning that must be acknowledged before continuing
        if (curTrial === nTraining - 1 && curResponse === null) {
            finishedTraining();
        }

        // Allow user to continue to the next trial
        if (canProceed) {
            $('#nextTrialMessage').show();
        }

        // Register which response was made
        if (event.which == 70) {
            curResponse = 'different';
            $('#option1box').css("background-color", "lightgrey");
            $('#option2box').css("background-color", "white");
        } else {
            curResponse = 'same';
            $('#option2box').css("background-color", "lightgrey");
            $('#option1box').css("background-color", "white");
        }

        // Display the answer if we're in the training phase
        if (training) {
            giveFeedback();
        }
    });

    // User wishes to continue to the next trial (pressed the "Space" key)
    $(document).bind("keydown.nextTrial", function (event) {
        // Check if they pressed the space bar and that they've responded
        // (and that they've acknowledged being done training)
        if (event.which == 32 && curResponse != null && canProceed) {
            $('#nextTrialMessage').hide();
            $('#feedback').hide();
            $('#option1box').css("background-color", "white");
            $('#option2box').css("background-color", "white");
            if (curTrial === nTraining - 1) {                   // If training has ended
                $("#sessionMode").html("Experiment segment")
            }
            trialDone();
        }
    });

    trialBegin();
}

function exportData() {
    $("#response").val(responses.join());
    $("#scene1").val(scene1s.join());
    $("#scene2").val(scene2s.join());
    $("#img1").val(img1s.join());
    $("#img2").val(img2s.join());
    $("#isCatch").val(isCatchs.join());
    $("#reactionTime").val(reactionTimes.join());
}

/* Setup/preloading code */

function getTrials(callback) {
    $.getJSON(rootPath + "assets/train_hit_set.json", function (dataTrain) {
        var trainTrials = dataTrain["train_hit_set"];
        for (var i = 0; i < trainTrials.length; i++) {
            trainTrials[i]["type"] = "train";
        }
        nTraining = trainTrials.length;
        $.getJSON(rootPath + "assets/catch_hit_set.json", function (dataCatch) {
            var catchTrials = dataCatch["catch_hit_set"];
            for (var i = 0; i < catchTrials.length; i++) {
                catchTrials[i]["type"] = "catch";
            }
            $.getJSON(rootPath + "assets/hit_sets.json", function (data) {
                expTrials = data["hit_sets"][hitId];
                for (var i = 0; i < expTrials.length; i++) {
                    expTrials[i]["type"] = "experiment";
                }

                // Mix the catch trials in with the experiment trials
                var trialsWithCatches = [];
                for (var iExp = 0, iCatch = 0; iExp < expTrials.length; iExp++) {
                  trialsWithCatches.push(expTrials[iExp]);
                  if (iExp % catchFreq == catchFreq - 1) {
                    trialsWithCatches.push(catchTrials[iCatch]);
                    iCatch++;
                  }
                }

                trials = trainTrials.concat(trialsWithCatches);
                callback();
            });
        });
    });
}

var imgCounter = 0;

function preloadStimuli(callback) {
    for (var i = 0; i < trials.length; i++) {
        preloadImg(trials[i], 1);
        preloadImg(trials[i], 2);
    }
    waitForStimuliToPreload(callback);
    console.log("Image preloading complete.");
}

function preloadImg(trial, imgNum) {
    let imgPath = rootPath + "images/" + trial["type"] + "/" +
        trial["scene" + imgNum] + "/" + trial["img" + imgNum] + ".jpg";
    loadImage(imgPath).then((img) => {
        console.log("Preloading:", img);
        trial["img" + imgNum + "Data"] = img;
        imgCounter++;
        console.log("Image preloading progress: " + Math.round(100 * (imgCounter / (2 * trials.length))) + "%");
    });
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        var img = new Image();
        img.onload = () => resolve(img);
        img.src = src;
    });
}

function waitForStimuliToPreload(callback) {
    if (imgCounter < (2 * trials.length)) {
        setTimeout(function () {
            waitForStimuliToPreload(callback)
        }, 24);
    } else {
        // load trial
        callback();
    }
}

$(document).ready(function () {
    $("#submitButton").hide();
    $("#sameDirectionImg").prop("src", rootPath + "assets/same_direction.png");
    $("#differentDirectionImg").prop("src", rootPath + "assets/different_direction.png");
    getTrials(function () {
        preloadStimuli(function () {
            $("#consent").click(function () {
                $("#startExperiment").click(function () {
                    startExperiment();
                });
            });
        });
    });
});

/* Utility functions */

function pad(num, size) {
    var s = num + "";
    while (s.length < size) s = "0" + s;
    return s;
}