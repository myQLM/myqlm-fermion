%define name qat-fermion

# Input
%{?!major:          %define major           0}
%{?!minor:          %define minor           0}
%{?!patchlevel:     %define patchlevel      0}
%{?!buildnumber:    %define buildnumber     0} 
%{?!branch:         %define branch          master} 

%if "%{branch}" == "rc" || "%{buildnumber}" == "0" 
%{?!version:        %define version         %{major}.%{minor}.%{patchlevel}} 
%else 
%{?!version:        %define version         %{major}.%{minor}.%{patchlevel}.%{buildnumber}} 
%endif 

%{?!rpm_release:    %define rpm_release     bull.0.0}
%{?!python_major:   %define python_major    3}
%{?!python_minor:   %define python_minor    6}
%{?!packager:       %define packager        noreply@eviden.com}
%{?!run_by_jenkins: %define run_by_jenkins  0}
%{?!platform:       %define platform        linux-x86_64}
%{?!python_distrib: %define python_distrib  linux-x86_64}

# Defines
%define python_version      %{python_major}.%{python_minor}
%define python_rpm          python%{python_major}%{python_minor}
%define workspace           %{getenv:WORKSPACE}

# Read location environment variables
%define target_bin_dir      /%{getenv:BIN_INSTALL_DIR}
%define target_lib_dir      /%{getenv:LIB_INSTALL_DIR}
%define target_headers_dir  /%{getenv:HEADERS_INSTALL_DIR}
%define target_thrift_dir   /%{getenv:THRIFT_INSTALL_DIR}
%define target_python_dir   /%{getenv:PYTHON_INSTALL_DIR}

%undefine __brp_mangle_shebangs

%if 0%{?srcrpm}
%undefine dist
%endif

# -------------------------------------------------------------------
#
# GLOBAL PACKAGE & SUB-PACKAGES DEFINITION
#
# -------------------------------------------------------------------
Name:           %{name}
Version:        %{version}
Release:        %{rpm_release}%{?dist}
Group:          Development/Libraries
Distribution:   QLM
Vendor:         Atos
License:        Bull S.A.S. proprietary : All rights reserved
BuildArch:      x86_64 
URL:            https://eviden.com/solutions/advanced-computing/quantum-computing

Source:         %{name}-%{version}.tar.gz
Source1:        qat.tar.gz


# -------------------------------------------------------------------
#
# MAIN PACKAGE DEFINITION
#
# -------------------------------------------------------------------
#%package main
Summary:  Quantum Application Toolset (QAT)
Requires: qat-core > 1.1.1
Provides: %{name}
AutoReq: no

%description
qat-fermion simulator. This package replaces the qat-dqs package.

# -------------------------------------------------------------------
#
# PREP
#
# -------------------------------------------------------------------
%prep
%setup -q
tar xfz %{SOURCE1} -C ..


# -------------------------------------------------------------------
#
# BUILD
#
# -------------------------------------------------------------------
%build
%if 0%{run_by_jenkins} == 0
QATDIR=%{_builddir}/qat
QAT_REPO_BASEDIR=%{_builddir}
RUNTIME_DIR=%{_builddir}/runtime
INSTALL_DIR=%{buildroot}

source /usr/local/bin/qatenv
# Restore artifacts
ARTIFACTS_DIR=$QAT_REPO_BASEDIR/artifacts
mkdir -p $RUNTIME_DIR
dependent_repos="$(get_dependencies.sh build %{name})"
while read -r dependent_repo; do
    [[ -n $dependent_repo ]] || continue
    tar xfz $ARTIFACTS_DIR/${dependent_repo}-*.tar.gz -C $RUNTIME_DIR
done <<< "$dependent_repos"
bldit -t debug -nd -ni -v ${name}
%endif


# -------------------------------------------------------------------
#
# INSTALL
#
# -------------------------------------------------------------------
%install
QATDIR=%{_builddir}/qat
QAT_REPO_BASEDIR=%{_builddir}
INSTALL_DIR=%{buildroot}

# Install it
%if 0%{run_by_jenkins} == 0
source /usr/local/bin/qatenv
bldit -t debug -nd -nc -nm ${name}

# Save artifact
ARTIFACTS_DIR=$QAT_REPO_BASEDIR/artifacts
mkdir -p $ARTIFACTS_DIR
tar cfz $ARTIFACTS_DIR/%{name}-%{version}-%{platform}-%{python_rpm}-%{python_distrib}.tar.gz -C $INSTALL_DIR .
%else
# Restore installed files
mkdir -p $INSTALL_DIR
tar xfz %{workspace}/%{name}-%{version}-%{platform}-%{python_rpm}-%{python_distrib}.tar.gz -C $INSTALL_DIR
%endif


# -------------------------------------------------------------------
#
# FILES
#
# -------------------------------------------------------------------
# Main package
%files
%defattr(-,root,root)
%{target_python_dir}/qat/*


# -------------------------------------------------------------------
#
# SCRIPTLETS
#
# -------------------------------------------------------------------
%pre

%post

%postun

%preun


# -------------------------------------------------------------------
#
# CHANGELOG
#
# -------------------------------------------------------------------
%changelog
* Sat May 04 2024 Jerome Pioux <jerome.pioux@eviden.com>
- Release 1.10 - Change files location to support Virtual Environments

-* Thu July 7 2022 Jerome Pioux <jerome.pioux@atos.net>
-- Initial release
