# Development Cycle

## Issue driven development

1. Choose an issue from the current milestone.
2. Create a branch (based on `main`) for that issue.
3. Develope code in dedicated branch.
4. Create Pull Request (PR) for merging issue branch back in to `main`.
5. Code in PR is reviewed and tested before merging.

!!! note "Only issues from current milestone will be merged!"

    While you can include work for an issue from a later milestone, it is at the discretion of the developer
    whether to change the issue to the current milestone for inclusion in the next release.
    
## The release cycle

!!! warning

    Only the repository administrators can perform these actions.

1. Create a milestone for the new release.
2. Create issues for tasks to be achieved during milestone.
3. Development is issue-driven, as described in the next section.
4. The release is ready when all issues in milestone have been addressed, and `main` test successfully.
5. The main branch is checked out at the last commit that completes the milestone.
6. The version is bumped and committed.
7. A release is created with a new tag set to the version commit.
8. The release is published on GitHub.

## Publishing a release

!!! warning

    For package administrators only!

Once a release has been published on GitHub, it can be published to PyPI.

1. Checkout the release tag with

   ```bash
   git fetch origin
   ```

   and then

   ```bash
   git checkout refs/tags/v.X.Y.Z
   ```

   where `v.X.Y.Z` is the version tag you want to publish.

2. Build the package with

   ```bash
   hatch build --clean
   ```

3. Publish the package with

   ```bash
   hatch publish
   ```

!!! example

    Eventually, the developer would like to have this automated by GitHub actions.
    Then the publishing of a release on PyPI would be directly tied to when the 
    release is published on GitHub.

### Setup for `hatch publish`

To run this command, you need to have setup a `.pypirc` file in your `HOME` directory with the following contents:

```toml
[com_pac]
  repository = https://upload.pypi.org/legacy/
  username = __token__
  password = <actual_token_goes_here>
```

The token is generated through the admin interface of PyPI for the `com_pac` package. 
