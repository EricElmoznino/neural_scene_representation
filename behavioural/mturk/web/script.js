let hitId = 0;

/* Parameters */
// let rootPath = "https://roi-disruption.s3.amazonaws.com/scene_behavioural/";
let rootPath = "";

/* Globals */
var trials = [];
var curTrial = 0;
var curResponse = null;
var nTraining;
var trialStartTime;
var training = true;
var canProceed = true;

/* Responses */
var responses = [];
var scene1s = [];
var scene2s = [];
var img1s = [];
var img2s = [];
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
        exportData();
        $("#trial").hide();
        $("#submitButton").show();
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
  });
}

function startExperiment() {
  $("#startExperiment").hide();
  $("#instructionsContainer").hide();
  $("#trial").show();

  // Click events

  // User has selected a response
  $("input[name=responseOption]").click(function () {
      $("#nextTrialButton").show();
      if (training) {
          $('#feedback').show();
          if (trials[curTrial]["scene1"] === trials[curTrial]["scene2"]) {
              $('#feedbackAnswer').html("Correct answer: SAME room");
          }
          else {
              $('#feedbackAnswer').html("Correct answer: Different rooms");
          }
      }
      if (curTrial === nTraining - 1 && curResponse === null) { // If this is the last training image, give a warning
          finishedTraining();
      }
      curResponse = $(this).val();
  });

  // User has clicked the button to continue to the next trial
  $("#nextTrial").click(function () {
    if (canProceed) {
        $("input[name=responseOption]").prop("checked", false);
        $("#nextTrialButton").hide();
        $('#feedback').hide();
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
  $("#reactionTime").val(reactionTimes.join());
}

/* Setup/preloading code */

function getTrials(callback) {
    $.getJSON(rootPath + "assets/train_hit_set.json", function (dataTrain) {
        var trainTrials = dataTrain["train_hit_set"];
        nTraining = trainTrials.length;
        $.getJSON(rootPath + "assets/hit_sets.json", function(data) {
            trials = data["hit_sets"][hitId];
            trials = trainTrials.concat(trials);
            callback();
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
    let imgPath = rootPath + "images/" + trial["scene"+imgNum] + "/" + trial["img"+imgNum]+ ".jpg";
      loadImage(imgPath).then((img) => {
          console.log("Preloading:", img);
          trial["img"+imgNum+"Data"] = img;
          imgCounter++;
          console.log("Image preloading progress: " + Math.round(100 * (imgCounter / (2 * trials.length))) + "%");
      });
}

function loadImage(src) {
    return new Promise((resolve, reject)=> {
        var img = new Image();
        img.onload = ()=> resolve(img);
        img.src = src;
    });
}

function waitForStimuliToPreload(callback) {
  if (imgCounter < (2 * trials.length)) {
      setTimeout(function() {waitForStimuliToPreload(callback)}, 24);
  } else {
      // load trial
      callback();
  }
}

$(document).ready(function() {
  $("#submitButton").hide();
  getTrials(function() {
    preloadStimuli(function(){
      $("#consent").click(function(){
          $("#startExperiment").click(function(){
              startExperiment();
          });
      });
  });
  });
});

/* Utility functions */

function pad(num, size) {
    var s = num+"";
    while (s.length < size) s = "0" + s;
    return s;
}