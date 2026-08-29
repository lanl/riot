<!--Provide a general summary of your changes in the title above, for
example "fix bug in mix.".  Please avoid
non-descriptive titles such as "Addresses issue #8576".-->

## PR Summary

<!--Please provide at least 1-2 sentences describing the pull request in
detail.  Why is this change required?  What problem does it solve?-->

<!--If it fixes an open issue, please link to the issue here.-->

## PR Checklist

<!-- Note that some of these check boxes may not apply to all pull requests -->

- [ ] Adds a test for any bugs fixed. Adds tests for new features.
- [ ] Format your changes by using the `scripts/format.sh` command or by using `@par-hermes format`
- [ ] Document any new features, update documentation for changes made.
- [ ] Make sure the copyright notice on any files you modified is up to date.
- [ ] LANL employees: make sure tests pass both on the github CI and on the re-git CI
- [ ] If ML was used, make sure to add a disclaimer at the top of a file indicating ML was used to assist in generating the file.
- [ ] If Agentic AI was used, have the AI generate a "proposed changes" markdown file and store it in the `plan_histories` folder, with a filename the same as the MR number.

If preparing for a new release, in addition please check the following:
- [ ] Update the version in cmake.
