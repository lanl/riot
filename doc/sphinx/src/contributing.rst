.. This file was made with the assistance of generative AI.

.. _contributing-doc:

Contributing
=============

If you have any trouble with the project, or are interested in
participating, please contact us by creating an issue on the GitHub
repository, or submit a pull request!

Pull request protocol
----------------------

There is a pull request template that will be auto-populated when you
submit a pull request. A pull request should have a summary of
changes. You should also add tests for bugs fixed or new features you
add.

Before a pull request will be merged, the code should be formatted. We
use clang-format for this, pinned to version 20. You can automatically
trigger ``clang-format`` in two ways: first you can run the script
``scripts/format.sh``; second you can create a comment on your MR
containing just the words ``@par-hermes format``. The former script
takes two CLI arguments that may be useful, ``CFM``, which can be set
to the path for your clang-format binary, and ``VERBOSE``, which if
set to ``1`` adds useful output. For example:

.. code-block:: bash

    CFM=clang-format-20 VERBOSE=1 ./util/scripts/format.sh

At least one maintainer must approve a pull request. Maintainers will,
at their discretion, tag/assign reviewers who are experts in relevant
parts of the code. If such an assignment is made, the relevant expert
must approve before the pull request can be merged.

Several sets of tests are triggered on a pull request: a static format
check, a docs build, and unit tests. These are run through GitHub's
CPU infrastructure. We have a second set of tests run on a wider set
of architectures, which we are not able to make public. Before it can
be merged, a maintainer must trigger the internal CI after review and
that second CI must pass.

Building the documentation
--------------------------

The documentation source is in ``doc/sphinx``. Create a Python virtual
environment and install the Sphinx dependencies:

.. code-block:: bash

   python3 -m venv docs_venv
   . docs_venv/bin/activate
   python -m pip install sphinx sphinx-rtd-theme sphinx-multiversion

Build the documentation for the current checkout with:

.. code-block:: bash

   cd doc/sphinx
   make html

Open ``doc/sphinx/_build/html/index.html`` in a browser to review the result.
To build documentation for all configured Git branches and tags, run ``make
multiversion`` instead.

AI-assisted coding
-------------------

``riot`` requires that if AI was used to assist in code
generation, a disclaimer must be made in a comment in the relevant
file. For example, you might add a comment like this one:

..code-block:: c++

  // This file was made in part with generative AI.

Also if agentic AI was used, please have your agent dump a
"proposed plan" markdown file in the ``plan_histories`` folder. This
provides an LLM-readable history of machine-generated changes and
helps disentangle human-made choices from machine-made ones. For
example, if you used codex or claude code, use a workflow like this
one:

1. Ask the agentic framework to propose a plan targeting your problem.
2. Tell it to dump the plan into a new file in ``plan_histories``
3. Iterate until you're happy with the code and submit an MR.
4. After submitting the MR, rename the new file to be prefixed by the MR number and commit it.

If you submit code to ``riot`` you own that code and you are
responsible for understanding it. If code is submitted that the author
does not understand, the author will be asked to resubmit a changeset
that they understand.

Finally, please be cognizant of reviewer time and effort. Agentic AI
can create changesets much faster than a human can review them. When
possible, please break up large changes and refactors into
human-parse-able chunks.

Expectations for code review
-----------------------------

From the perspective of the contributor
````````````````````````````````````````

Code review is an integral part of the development process
for ``riot``. You can expect at least one, perhaps many,
core developers to read your code and offer suggestions.
You should treat this much like scientific or academic peer review.
You should listen to suggestions but also feel entitled to push back
if you believe the suggestions or comments are incorrect or
are requesting too much effort.

Reviewers may offer conflicting advice, if this is the case, it's an
opportunity to open a discussion and communally arrive at a good
approach. You should feel empowered to argue for which of the
conflicting solutions you prefer or to suggest a compromise. If you
don't feel strongly, that's fine too, but it's best to say so to keep
the lines of communication open.

Big contributions may be difficult to review in one piece and you may
be requested to split your pull request into two or more separate
contributions. You may also receive many "nitpicky" comments about
code style or structure. These comments help keep a broad codebase,
with many contributors uniform in style and maintainable with
consistent expectations across the code base. While there is no
formal style guide for now, the regular contributors have a sense for
the broad style of the project. You should take these stylistic and
"nitpicky" suggestions seriously, but you should also feel free to
push back.

As with any creative endeavor, we put a lot of ourselves into our
code. It can be painful to receive criticism on your contribution and
easy to take it personally. While you should resist the urge to take
offense, it is also partly code reviewer's responsibility to create a
constructive environment, as discussed below.

Expectations of code reviewers
````````````````````````````````

A good code review builds a contribution up, rather than tearing it
down. Here are a few rules to keep code reviews constructive and
congenial:

* You should take the time needed to review a contribution and offer
  meaningful advice. Unless a contribution is very small, limit
  the times you simply click "approve" with a "looks good to me."

* You should keep your comments constructive. For example, rather than
  saying "this pattern is bad," try saying "at this point, you may
  want to try this other pattern."

* Avoid language that can be misconstrued, even if it's common
  notation in the community. For example, avoid phrases like "code
  smell."

* Explain why you make a suggestion. In addition to saying "try X
  instead of Y" explain why you like pattern X more than pattern Y.

* A contributor may push back on your suggestion. Be open to the
  possibility that you're either asking too much or are incorrect in
  this instance. Code review is an opportunity for everyone to learn.

* Don't just highlight what you don't like. Also highlight the parts
  of the pull request you do like and thank the contributor for their
  effort.

General principle for everyone
```````````````````````````````

It's hard to convey tone in text correspondence. Try to read what
others write favorably and try to write in such a way that your tone
can't be mis-interpreted as malicious.

Interwoven Dependencies
------------------------

``riot`` depends on several other open-source, Los Alamos maintained,
projects. In particular, ``singularity-eos``, ``singularity-opac``,
``spiner`` and ``ports-of-call``. If you have issues with these
projects, ideally submit issues on the relevant GitHub pages. However,
if you can't figure out where an issue belongs, no big deal. Submit
where you can and we'll engage with you to figure out how to proceed.


Notes for Contributors on navigating/developing code features
-------------------------------------------------------------

Performance portability concerns
`````````````````````````````````

``riot`` is performance portable, meaning it is designed to
run not only on CPUs, but GPUs from a variety of manufacturers,
powered by a variety of device-side development tools such as Cuda,
OpenMP, and OpenACC. This implies several constraints on code
style. Here we briefly discuss a few things one should be aware of.

* **`portability decorators:** Functions that should be run on device
  needs to be decorated with one of the following macros:
  ``KOKKOS_FUNCTION``, ``KOKKOS_INLINE_FUNCTION``,
  ``KOKKOS_FORCEINLINE_FUNCTION``. These macros are imported from the
  Kokkos library and resolve to the appropriate decorations for a
  given device-side backend such as Cuda so the code compiles
  correctly. Code that doesn't need to run on device does not need
  these decorations.

* **Relocatable device code:** It is common in C++ to split code
  between a header file and an implementation file. Functionality that
  is to be called from within loops run on device should not be split
  in this way. Not all accelerator languages support this and the ones
  that do take a performance hit. Instead implement that functionality
  only in a header file and decorate it with
  ``KOKKOS_INLINE_FUNCTION`` or ``KOKKOS_FORCEINLINE_FUNCTION``.

* **Host and device pointers:** Usually accelerators have different
  memory spaces than the CPU they are attached to. So you need to be
  aware that data needs to be copied to an accelerator device to be
  used. If it is not properly copied, the code will likely crash with
  a segfault. In general scalar data such as a single variable (e.g.,
  ``int x``) can be easily and automatically copied to device and you
  don't need to worry about managing it. Arrays and pointers, however,
  are a different story. If you create an array or point to some
  memory on CPU, then you are pointing to a location in memory on your
  CPU. If you try to access it from your accelerator, your code will
  not behave properly. You need to manually copy data from host to
  device in this case.

* **Real:** The ``Real`` datatype is either a single precision or
  double precision floating point number, depending on how
  ``Parthenon`` is configured. For most floating point numbers use
  the ``Real`` type. However, be conscious that sometimes you will
  specifically need a single or double precision number, in which case
  you should specify the type as built into the language.


How to Make a Release
----------------------

``riot`` uses *date-based*. A version is written as ``yyyy.mm.dd``. To
make a new release, first make a new pull request where you change the
version number in the ``project`` field of the of the top-level
``CmakeLists.txt`` file. Typically the branch for this merge request
should be called ``v[release number]-rc`` for "release candidate."
Make sure that the full test suite passes for this PR.

After that pull request is merged, go to the ``releases`` tab on the
right sidebar on GitHub, and draft a new release. Set the tag to
``[release number]``. You can let github automatically draft a release
note by summarizing MRs.

Updating regression-test gold files
------------------------------------

Regression-test reference data is distributed separately from the source tree
as a GitHub Release asset. This procedure is for maintainers updating that
data. The GitHub Release is created manually so that only a maintainer with
repository release permissions needs GitHub credentials.

#. From ``tst/scripts/gold/files``, bundle the updated gold files and update
   the CMake version and checksum pin. Optionally provide the directory from a
   test run to refresh existing gold files:

   .. code-block:: bash

      ./bundle_goldfiles.sh --update-cmake [test-run-directory]

   This produces ``riot_regression_gold_<version>.tgz`` and updates
   ``RIOT_REGRESSION_GOLD_VER`` and ``RIOT_REGRESSION_GOLD_HASH`` in the
   top-level ``CMakeLists.txt``. Do not rename or regenerate this archive after
   running the command, because its exact contents and filename determine the
   pinned checksum.

#. On GitHub, create a release tagged ``regression-gold-<version>`` and upload
   that exact ``riot_regression_gold_<version>.tgz`` archive as its release
   asset. Publish the release only after confirming the tag and asset filename.

#. Commit the updated top-level ``CMakeLists.txt``,
   ``tst/scripts/gold/files/current_version``, and
   ``tst/scripts/gold/files/README.md``. The archive and extracted gold files
   are release artifacts and should not be committed to the source repository.

#. Verify the release from a clean checkout by configuring with
   ``-DRIOT_ENABLE_REGRESSION_TESTS=ON``. CMake should download the asset and
   verify its SHA-512 checksum before extracting it.

Continuous Integration
----------------------

``riot`` has two continuous integration (CI) systems. A public
facing one via GitHub actions and an LANL internal one through a GitLab
instance. The GitHub actions are configured via the files located in the
``.github/workflows`` subdirectory.

Our GitLab CI is configured via the ``.gitlab-ci.yml`` file. To
trigger the GitLab CI runs, you need to have access to our internal
GitLab instance, push your branch to this second Git repository, and
create a draft GitLab merge request (MR). Each GitLab MR will launch a
pipeline with multiple jobs on various clusters.

Be aware that the CI on the system runs sequentially and the number of
concurrent jobs per user is limited. You may wish to cancel an old run
if you no longer need the results and want your most recent run to finish.

Setting git to automatically push to our CI system
````````````````````````````````````````````````````

If you would like to have git automatically push to the CI system when
you type ``git push``, you can do so. The following procedure is
recommended:

.. code-block:: bash

   git remote add ci <git ssh path to ci repo>
   git remote add all git@github.com:lanl/riot.git
   git remote set-url --add --push all git@github.com:lanl/riot.git
   git remote set-url --add --push all <git ssh path to ci repo>
   git config remote.pushDefault all

With these changes,

.. code-block:: bash

   git pull
   # pulls from github

   git push
   # pushes to both github and the CI machine via "all"

   git push origin
   # pushes only to github

   git push ci
   # pushes only to the CI machine
