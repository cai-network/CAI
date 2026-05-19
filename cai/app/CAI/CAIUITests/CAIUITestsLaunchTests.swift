// SPDX-FileCopyrightText: 2025 cai Technologies Ltd
// SPDX-FileCopyrightText: 2026 CAI contributors
// SPDX-License-Identifier: Apache-2.0
//
//  CAIUITestsLaunchTests.swift
//  CAIUITests
//
//  Created by Sami Khan on 2025-11-22.
//

import XCTest

final class CAIUITestsLaunchTests: XCTestCase {

    override class var runsForEachTargetApplicationUIConfiguration: Bool {
        true
    }

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    @MainActor
    func testLaunch() throws {
        let app = XCUIApplication()
        app.launch()

        // Insert steps here to perform after app launch but before taking a screenshot,
        // such as logging into a test account or navigating somewhere in the app

        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "Launch Screen"
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
