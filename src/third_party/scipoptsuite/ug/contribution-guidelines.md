# Setup

The first way to get started developing [UG](http://ug.zib.de/) is to simply clone the [repository](https://git.zib.de/integer/ug.git).

# Contributing

Before starting to work on a code change, go through the following list:

- Check the list of issues and merge requests to see if somebody else is already working on this or not.
- Select what branch to base your contribution on, bugfixes should be branched off of the current bugfix branch, where new features should be based on the master branch.
- How atomic is your feature? If possible, try to break down large changes into a sequence of smaller changes that can be reviewed and merged sequentially.
- Large features should start with a Milestone. In the Milestone, several, discrete chunks of work are elaborated as Issues. When creating these issues, make the issue description as detailed as possible, to prompt further technical discussion.
- Small features can start with a single issue or also directly with a merge request. In any case you need to provide a description such that other developers can comment.

# Merge Requests

When fixing an issue, start your new branch name with the issue number and short description and branch off of the current bugfix branch.

When developing a feature, start your new branch name with a short description and branch off of master.

You can then already create a merge request to see how the continuous integration is doing.
Please also select the `code review` as a merge request template and inform your reviewer about the status of your code.
Now the continuous integration server is going to build all tests whenever you push a change to your feature branch.
It will try the merge before and run the tests on the hypothetical result of the merge.
You can also see the output of the tests directly from your merge request.

Don't forget to assign a reviewer and an assignee to the newly-created Merge Request.

After work on your branch is complete, ensure that your branch is up-to-date with all of the changes from your base branch by merging the base branch into your development branch.

When the new feature is reviewed and the build is green, the assignee of the merge request can push the button to merge.

For those unfamiliar with creating Merge Requests in GitLab, check out [this helpful documentation](http://doc.gitlab.com/ee/gitlab-basics/add-merge-request.html) that explains the steps.

A Merge Request can be opened at any point in your development cycle: beginning, middle or end of development
-  The benefits of opening one early is that builds and checks will happen automatically every time that you push your branch.
-  If you add `WIP:` to the beginning of the Merge Request you can deactivate the CI builds temporarily.
   This may be useful at the beginning, when you know that code does not build correctly, yet.
-  The orange "Close" button closes the Merge Request without merging it.
   Closing a Merge Request is effectively the same as abandoning a branch, and said branch should be subsequently deleted.
-  Branches can safely be deleted after a Merge Request is merged since the history will be preserved in the destination branch.

To keep master up-to-date, the bugfix branch is regularly merged into master.
You can help by doing this after successfully merging your bugfix merge request.



